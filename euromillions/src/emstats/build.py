"""Reconcile the sources into one validated draw table.

Every disagreement between sources, and every repair, is written to
data/processed/reconciliation_log.csv. Anything the rules below cannot
resolve stops the build rather than being silently picked.
"""
import json

import numpy as np
import pandas as pd

from . import config as C

MAINS = ["n1", "n2", "n3", "n4", "n5"]
STARS = ["s1", "s2"]


class DataError(RuntimeError):
    pass


def _canon(df):
    """Sort balls within each draw so sources compare like for like."""
    out = df.copy()
    out[MAINS] = np.sort(out[MAINS].to_numpy(), axis=1)
    out[STARS] = np.sort(out[STARS].to_numpy(), axis=1)
    out["date"] = pd.to_datetime(out["date"])
    return out[["date"] + MAINS + STARS].astype({c: int for c in MAINS + STARS})


def _fmt(row):
    if row is None:
        return ""
    return " ".join(str(int(row[c])) for c in MAINS) + " | " + " ".join(str(int(row[c])) for c in STARS)


def load_primary():
    return _canon(pd.read_csv(C.PRIMARY_FILE))


def load_secondary():
    frames = [pd.read_csv(f, sep=r"\s+", dtype=str) for f in sorted(C.SECONDARY_DIR.glob("*.txt"))]
    df = pd.concat(frames, ignore_index=True)
    df.columns = ["date"] + MAINS + STARS
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df[MAINS + STARS] = df[MAINS + STARS].astype(int)
    df = _canon(df).sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise DataError("secondary source has duplicate dates")
    return df


def expected_schedule(last_date):
    days = pd.date_range(C.FIRST_DRAW, last_date, freq="D")
    fri = days.dayofweek == 4
    tue = (days.dayofweek == 1) & (days >= C.TUESDAY_START)
    return pd.DatetimeIndex(days[fri | tue])


def era_of(dates):
    labels = np.full(len(dates), C.ERAS[0][0], dtype=object)
    pools = np.full(len(dates), C.ERAS[0][2])
    for label, start, pool in C.ERAS[1:]:
        m = dates >= start
        labels[m], pools[m] = label, pool
    return labels, pools


def _same(a, b):
    return all(int(a[c]) == int(b[c]) for c in MAINS + STARS)


def reconcile(primary, secondary, manual):
    log = []
    df = primary.copy().set_index("date")

    # 1. Documented manual corrections (each asserts the numbers it expects).
    for _, m in manual.iterrows():
        d = pd.Timestamp(m["date"])
        rec = {c: int(m[c]) for c in MAINS + STARS}
        canon = _canon(pd.DataFrame([{"date": d, **rec}])).iloc[0]
        if m["action"] in ("redate", "drop"):
            if d not in df.index or not _same(df.loc[d], canon):
                raise DataError(f"manual {m['action']} {d.date()}: primary row missing or numbers changed")
            before = _fmt(df.loc[d])
            df = df.drop(index=d)
            if m["action"] == "redate":
                nd = pd.Timestamp(m["new_date"])
                if nd in df.index:
                    raise DataError(f"manual redate target {nd.date()} already present")
                df.loc[nd] = canon[MAINS + STARS]
                res = f"re-dated to {nd.date()}"
            else:
                res = "dropped"
            log.append(dict(date=d.date(), issue=f"primary: off-schedule row ({d.day_name()})",
                            primary=before, secondary="", other_source=m["source"] if isinstance(m["source"], str) else "",
                            resolution=res, note=m["note"]))
        elif m["action"] == "add":
            if d in df.index:
                if _same(df.loc[d], canon):
                    continue  # primary has since been fixed upstream; nothing to do
                raise DataError(f"manual add {d.date()}: primary now has different numbers")
            df.loc[d] = canon[MAINS + STARS]
            log.append(dict(date=d.date(), issue="primary: scheduled draw missing",
                            primary="", secondary="", other_source=m["source"],
                            resolution="added from other source", note=m["note"]))
        else:
            raise DataError(f"unknown manual action {m['action']}")
    df = df.sort_index()

    sched = set(expected_schedule(df.index.max()))
    sec = secondary.set_index("date")

    # 2. Off-schedule rows that duplicate a neighbouring draw exactly.
    for d in [x for x in df.index if x not in sched]:
        near = df[(abs((df.index - d).days) <= 3) & (df.index != d)]
        dup = [x for x in near.index if _same(near.loc[x], df.loc[d])]
        if dup:
            log.append(dict(date=d.date(), issue=f"primary: off-schedule row ({d.day_name()}) duplicates {dup[0].date()}",
                            primary=_fmt(df.loc[d]), secondary="", other_source="", resolution="dropped as duplicate",
                            note="identical numbers to the adjacent scheduled draw"))
            df = df.drop(index=d)

    # 3. Off-schedule rows that the secondary source places on a nearby scheduled date.
    for d in [x for x in df.index if x not in sched]:
        cands = [x for x in sec.index if abs((x - d).days) <= 3 and x in sched
                 and x not in df.index and _same(sec.loc[x], df.loc[d])]
        if len(cands) == 1:
            nd = cands[0]
            log.append(dict(date=d.date(), issue=f"primary: off-schedule row ({d.day_name()})",
                            primary=_fmt(df.loc[d]), secondary=f"{nd.date()}: {_fmt(sec.loc[nd])}",
                            other_source="", resolution=f"re-dated to {nd.date()} (secondary agrees)",
                            note="date typo in primary"))
            df.loc[nd] = df.loc[d]
            df = df.drop(index=d)
    df = df.sort_index()

    # 4. Overlap comparison with the secondary source.
    lo, hi = sec.index.min(), sec.index.max()
    n_overlap_agree = 0
    for d in sec.index:
        if d in df.index:
            if _same(df.loc[d], sec.loc[d]):
                n_overlap_agree += 1
            else:
                log.append(dict(date=d.date(), issue="sources disagree on numbers",
                                primary=_fmt(df.loc[d]), secondary=_fmt(sec.loc[d]), other_source="",
                                resolution="UNRESOLVED", note="add a manual correction with evidence"))
        else:
            df.loc[d] = sec.loc[d]
            log.append(dict(date=d.date(), issue="primary: scheduled draw missing",
                            primary="", secondary=_fmt(sec.loc[d]), other_source="",
                            resolution="added from secondary (single source)", note=""))
    for d in df.index[(df.index >= lo) & (df.index <= hi)]:
        if d not in sec.index:
            log.append(dict(date=d.date(), issue="secondary: draw missing",
                            primary=_fmt(df.loc[d]), secondary="", other_source="",
                            resolution="kept primary (single source)", note=""))
    df = df.sort_index()
    log = pd.DataFrame(log, columns=["date", "issue", "primary", "secondary", "other_source", "resolution", "note"])
    df = df.astype(int).reset_index().rename(columns={"index": "date"})
    return df, log, n_overlap_agree, (lo, hi)


def validate(df, log):
    errors = []
    if (log["resolution"] == "UNRESOLVED").any():
        errors.append(f"{(log['resolution'] == 'UNRESOLVED').sum()} unresolved source conflicts")
    if df["date"].duplicated().any():
        errors.append("duplicate dates")
    mains = df[MAINS].to_numpy()
    stars = df[STARS].to_numpy()
    if mains.min() < 1 or mains.max() > C.MAIN_POOL:
        errors.append("main ball out of 1-50")
    if (np.diff(mains, axis=1) == 0).any():
        errors.append("repeated main ball within a draw")
    if (np.diff(stars, axis=1) == 0).any():
        errors.append("repeated Lucky Star within a draw")
    if df[MAINS + STARS].duplicated().any():
        errors.append("identical full result on two dates (possible duplicate row)")
    if (stars.min(axis=1) < 1).any() or (stars.max(axis=1) > df["star_pool"].to_numpy()).any():
        errors.append("Lucky Star outside its era's range")
    sched = expected_schedule(df["date"].max())
    missing = sorted(set(sched) - set(df["date"]))
    extra = sorted(set(df["date"]) - set(sched))
    if missing:
        errors.append(f"missing scheduled draws: {[d.date().isoformat() for d in missing]}")
    if extra:
        errors.append(f"off-schedule draws: {[d.date().isoformat() for d in extra]}")
    if errors:
        raise DataError("validation failed:\n  " + "\n  ".join(errors))


def era_evidence(df):
    """Empirical check of the rule-change dates: first/last appearance of each star value."""
    rows = []
    for v in range(1, 13):
        hit = df[(df["s1"] == v) | (df["s2"] == v)]["date"]
        rows.append(dict(star=v, first_seen=hit.min().date(), last_seen=hit.max().date(), draws=len(hit)))
    return pd.DataFrame(rows)


def build():
    primary, secondary = load_primary(), load_secondary()
    manual = pd.read_csv(C.MANUAL_CORRECTIONS)
    df, log, n_agree, (lo, hi) = reconcile(primary, secondary, manual)
    df["era"], df["star_pool"] = era_of(df["date"])
    validate(df, log)
    df.insert(0, "draw_index", np.arange(len(df)))
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.PROCESSED / "draws.csv", index=False, date_format="%Y-%m-%d")
    log.to_csv(C.PROCESSED / "reconciliation_log.csv", index=False)
    ev = era_evidence(df)
    ev.to_csv(C.TABLES / "star_value_first_seen.csv", index=False)
    summary = dict(
        n_draws=len(df), first=str(df["date"].min().date()), last=str(df["date"].max().date()),
        n_primary_rows=len(primary), n_secondary_rows=len(secondary),
        secondary_range=[str(lo.date()), str(hi.date())], overlap_agree=n_agree,
        n_log=len(log), draws_per_era=df["era"].value_counts().reindex([e[0] for e in C.ERAS]).to_dict(),
        main_min=int(df[MAINS].min().min()), main_max=int(df[MAINS].max().max()),
    )
    (C.PROCESSED / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"built {len(df)} draws {summary['first']} to {summary['last']}; "
          f"{n_agree} overlap draws agree; {len(log)} reconciliation entries")
    return df, log, summary


def load_draws():
    return pd.read_csv(C.PROCESSED / "draws.csv", parse_dates=["date"])
