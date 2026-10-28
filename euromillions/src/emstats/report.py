"""Write report.md from the analysis results. The conclusion is derived from
the numbers, so a refresh with new draws rewrites it consistently."""
import json

import numpy as np
import pandas as pd

from . import config as C

FIG = "outputs/figures"


def fp(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def f2(x, d=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def md_table(df, cols, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, r in df.iterrows():
        cells = (str(c(r)) if callable(c) else str(r[c]) for c in cols)
        out.append("| " + " | ".join(x.replace("|", "\\|") for x in cells) + " |")
    return "\n".join(out)


def conclusion(res, bt, summary):
    tests = res["tests"]
    flagged_global = tests[tests["p_bh_family"] < C.ALPHA]
    main_sig = res["main_table"][res["main_table"]["significant_after_correction"]]
    star_sig = res["star_table"][res["star_table"]["significant_after_correction"]]
    pair_sig = res["pair_table"][res["pair_table"]["p_bh"] < C.ALPHA]
    gap_sig = res["gap_table"][res["gap_table"]["p_bh"] < C.ALPHA]
    tab = bt["table"].set_index("strategy")
    others = tab.drop(index="uniform")
    winners = others[(others["logloss_vs_uniform"] < 0) & (others["p_perm_logloss"] < C.ALPHA)
                     & (others["p_beats_uniform_logloss"] < C.ALPHA)]
    any_below = others[others["logloss_vs_uniform"] < 0]
    hit_winners = others[others["p_hits_exact"] < C.ALPHA]
    nominal = tests.sort_values("p_used").iloc[0]

    a1_clean = flagged_global.empty and main_sig.empty and star_sig.empty and pair_sig.empty \
        and gap_sig.empty and res["autocorr_bh_sig"] == 0
    s = []
    if a1_clean:
        s.append(f"No. Across {summary['n_draws']:,} draws ({summary['first']} to {summary['last']}) the "
                 "EuroMillions history is indistinguishable from a fair random draw: no main ball, Lucky Star, "
                 "pair, gap pattern or lag relationship survives correction for multiple testing.")
    else:
        bits = []
        if not flagged_global.empty:
            bits.append(f"{len(flagged_global)} global test(s) significant after BH correction "
                        f"({'; '.join(flagged_global['test'])})")
        for name, t in [("main ball", main_sig), ("Lucky Star", star_sig), ("pair", pair_sig), ("gap", gap_sig)]:
            if not t.empty:
                bits.append(f"{len(t)} {name} result(s) significant after correction")
        if res["autocorr_bh_sig"]:
            bits.append(f"{res['autocorr_bh_sig']} autocorrelation(s) significant after BH")
        s.append(f"Across {summary['n_draws']:,} draws ({summary['first']} to {summary['last']}) the history "
                 f"is broadly consistent with a fair draw, but the following survived correction: {'; '.join(bits)}. "
                 "Treat these as leads to re-test on future draws, not as an edge.")
    s.append(f"The smallest p-value among the {len(tests)} global tests is {fp(nominal['p_used'])} "
             f"({nominal['test']}), which becomes {fp(nominal['p_bh_family'])} once you account for running "
             f"{len(tests)} tests; that is what chance alone produces.")
    m = res["main_table"].sort_values("p_raw").iloc[0]
    s.append(f"The nearest miss among single numbers is ball {int(m['number'])}, drawn {int(m['count'])} times "
             f"against {m['expected']:.0f} expected (raw p = {fp(m['p_raw'])}, {fp(m['p_bh'])} after correction); "
             "one number in 50 looking like that is ordinary.")
    if winners.empty:
        best = others.sort_values("logloss").iloc[0]
        tail = (f"{len(any_below)} of them scored a marginally lower log-loss than uniform, but not beyond what "
                "shuffled histories produce. " if len(any_below) else "None of them even matched the uniform "
                "baseline's log-loss. ")
        s.append(f"In the walk-forward backtest over {bt['n_test']:,} draws, no strategy (hot, hot-recent, "
                 f"overdue or logistic regression) beat the uniform 5/50 baseline. {tail}"
                 f"The best was {best['label'].lower()} at {best['logloss_vs_uniform'] * 1000:+.3f} "
                 f"millinats per number versus uniform (permutation p = {fp(best['p_perm_logloss'])}).")
    else:
        s.append(f"In the walk-forward backtest, {', '.join(winners['label'])} beat uniform beyond the "
                 "permutation null. Check whether this survives on draws after "
                 f"{summary['last']} before believing it; with {len(others)} strategies tested, one false "
                 "positive at 5% is unsurprising.")
    if hit_winners.empty:
        s.append(f"Top-5 picks hit {others['mean_hits_top5'].min():.3f}–{others['mean_hits_top5'].max():.3f} "
                 "main balls per draw against 0.5 for random picks, all within chance.")
    s.append("Past results carry no usable predictive signal. The only lever a player controls is not "
             "the odds but how many people they would share a jackpot with.")
    return " ".join(s)


def write(res, bt, summary, recon_log):
    tests = res["tests"]
    tab = bt["table"]
    lines = ["# EuroMillions: do past draws predict future draws?", ""]
    lines += [f"_Data: {summary['n_draws']:,} draws, {summary['first']} to {summary['last']}. "
              f"Generated by `python run_all.py`. Seed {C.SEED}; {summary['n_null_sims']:,} Monte Carlo histories; "
              f"{summary['n_permutations']:,} draw-order permutations._", ""]
    lines += ["## Conclusion", "", conclusion(res, bt, summary), ""]

    # ---- data
    era_counts = summary["draws_per_era"]
    lines += ["## Data", "",
              f"- **Primary source:** [daowa89/lottery-archive]({C.PRIMARY_URL.replace('raw.githubusercontent.com', 'github.com').replace('/main/', '/blob/main/')}) "
              "`eu/euromillions/results.csv`, a GitHub-Actions-maintained archive that pulls from win2day.at "
              "(Österreichische Lotterien, an official EuroMillions operator). Full history, updated each draw day. "
              f"Saved as `data/raw/{C.PRIMARY_FILE.name}` ({summary['n_primary_rows']:,} rows as fetched).",
              f"- **Secondary source (cross-check):** result files bundled in the `{C.SECONDARY_PACKAGE}=={C.SECONDARY_VERSION}` "
              f"PyPI package, covering {summary['secondary_range'][0]} to {summary['secondary_range'][1]} "
              f"({summary['n_secondary_rows']:,} draws). Saved under `data/raw/pyeuromil/`.",
              "- **Third source, for gaps outside the secondary's coverage only:** individual result pages on "
              "euro-millions.com (cited per row in `data/manual_corrections.csv`).",
              "- Official operator sites (FDJ, National Lottery) were not reachable from the build environment, so they "
              "could not be used directly.",
              "",
              f"**Reconciliation.** Where both main sources cover the same date they agree on all "
              f"{summary['overlap_agree']:,} draws. The primary file has {len(recon_log)} problems, all listed below "
              "and in `data/processed/reconciliation_log.csv`. Nothing was resolved silently; a new unresolved "
              "conflict stops the build.", ""]
    lines.append(md_table(recon_log, ["date", "issue", lambda r: r["primary"] or "–",
                                       lambda r: r["secondary"] or "–", "resolution",
                                       lambda r: r["other_source"] or "–"],
                          ["Date in primary", "Problem", "Primary row", "Secondary row", "Resolution", "Other evidence"]))
    lines += ["",
              "**Validation (all pass):** no duplicate dates or results; five distinct main balls in 1–50 and two "
              "distinct stars in range for every draw; every Friday from 13 Feb 2004 and every Tuesday from "
              f"10 May 2011 to {summary['last']} is present, with no off-schedule draws. Main balls observed: "
              f"{summary['main_min']}–{summary['main_max']}.", "",
              "**Lucky Star eras.** Tagged by date and confirmed against the data (`outputs/tables/star_value_first_seen.csv`):", "",
              "| Era | Pool | First draw | Draws | Evidence in data |", "|---|---|---|---|---|",
              f"| 1-9 | 9 | 13 Feb 2004 | {era_counts['1-9']} | no star above 9 before 10 May 2011 |",
              f"| 1-11 | 11 | 10 May 2011 (first Tuesday draw) | {era_counts['1-11']} | stars 10 and 11 first appear May–Jun 2011; no 12 before 27 Sep 2016 |",
              f"| 1-12 | 12 | 27 Sep 2016 | {era_counts['1-12']} | star 12 first drawn on 27 Sep 2016 |",
              "", "Star analyses are run within each era only.", ""]

    # ---- analysis 1
    lines += ["## Analysis 1: is the history consistent with randomness?", "",
              "Each global statistic is compared with its exact or asymptotic distribution where one exists, and "
              f"always with {summary['n_null_sims']:,} simulated fair histories of the same length. The frequency chi-square "
              "is corrected for drawing without replacement (the plain Pearson statistic understates variation by "
              "a factor (1−p)·N/(N−1)). The last column applies Benjamini-Hochberg across the whole family of global tests.", ""]
    lines.append(md_table(tests, ["test", lambda r: f2(r["statistic"]), lambda r: "–" if pd.isna(r["df"]) else int(r["df"]),
                                  lambda r: fp(r["p_analytic"]), lambda r: fp(r["p_monte_carlo"]),
                                  lambda r: fp(r["p_bh_family"]), lambda r: r["note"] or ""],
                          ["Test", "Statistic", "df", "p (analytic)", "p (Monte Carlo)", "p (BH, family)", "Note"]))
    mt = res["main_table"]
    lines += ["", f"![Main ball frequencies]({FIG}/fig1_main_frequency.png)", "",
              "Bars outside the pale outer band would be significant after Bonferroni correction (and would be drawn in red); "
              f"{int(mt['significant_after_correction'].sum())} are.", "",
              "**Most extreme main balls (two-sided exact binomial):**", ""]
    lines.append(md_table(mt.sort_values("p_raw").head(6),
                          ["number", "count", lambda r: f2(r["expected"], 1), lambda r: f"{r['ci_low']:.0f}–{r['ci_high']:.0f}",
                           lambda r: fp(r["p_raw"]), lambda r: fp(r["p_bh"]), lambda r: fp(r["p_bonferroni"])],
                          ["Ball", "Drawn", "Expected", "95% CI (count)", "p raw", "p BH", "p Bonferroni"]))
    st = res["star_table"]
    lines += ["", f"![Lucky Star frequencies]({FIG}/fig2_star_frequency.png)", "",
              "**Most extreme Lucky Stars (within era):**", ""]
    lines.append(md_table(st.sort_values("p_raw").head(6),
                          ["era", "number", "count", lambda r: f2(r["expected"], 1), lambda r: fp(r["p_raw"]),
                           lambda r: fp(r["p_bh"]), lambda r: fp(r["p_bonferroni"])],
                          ["Era", "Star", "Drawn", "Expected", "p raw", "p BH (within era)", "p Bonferroni"]))
    gt = res["gap_table"]
    lines += ["", f"![Gap distributions]({FIG}/fig3_gaps.png)", "",
              f"Gaps are measured in draws (1 = drawn again in the next draw). Under a fair draw they are "
              f"geometric with p = 0.1 and mean 10. Per-number gap tests: {int((gt['p_raw'] < 0.05).sum())} of 50 "
              f"nominally significant, {int((gt['p_bh'] < C.ALPHA).sum())} after BH. Longest gap in the history: "
              f"{int(gt['max_gap'].max())} draws (ball {int(gt.loc[gt['max_gap'].idxmax(), 'number'])}).", "",
              f"![Serial overlap]({FIG}/fig4_serial_overlap.png)", ""]
    lt = res["lag_table"]
    lines.append(md_table(lt, [lambda r: int(r["lag"]), lambda r: f"{r['mean_overlap']:.3f}", lambda r: f2(r["z"]), lambda r: fp(r["p_raw"]),
                               lambda r: fp(r["p_bh"]), lambda r: fp(r["dist_p"])],
                          ["Lag", "Mean shared balls", "z", "p", "p BH", "p (full overlap distribution)"]))
    pt = res["pair_table"]
    lines += ["", f"![Pairs]({FIG}/fig5_pairs.png)", "",
              f"Each of the 1,225 pairs should appear together in {res['p_pair'] * 100:.3f}% of draws "
              f"({pt['expected'].iloc[0]:.1f} times). {int((pt['p_raw'] < 0.05).sum())} pairs are nominally significant "
              f"at 5% (about {0.05 * 1225:.0f} expected by chance); {int((pt['p_bh'] < C.ALPHA).sum())} survive BH and "
              f"{int((pt['p_bonferroni'] < C.ALPHA).sum())} survive Bonferroni.", "",
              "**Most extreme pairs:**", ""]
    lines.append(md_table(pt.head(5), [lambda r: f"{int(r['a'])}–{int(r['b'])}", lambda r: int(r["count"]), lambda r: f2(r["expected"], 1),
                                       lambda r: fp(r["p_raw"]), lambda r: fp(r["p_bh"])],
                          ["Pair", "Together", "Expected", "p raw", "p BH"]))
    lines += ["", f"![Sum and odd/even]({FIG}/fig6_sum_oddeven.png)", "",
              "The sum distribution is computed exactly (all 2,118,760 combinations, by dynamic programming); the "
              "odd count is hypergeometric(50, 25, 5).", ""]

    # ---- analysis 2
    lines += ["## Analysis 2: does any strategy beat random?", "",
              f"Walk-forward over draws {C.BURN_IN + 1} to {summary['n_draws']} ({bt['test_start']} onwards, "
              f"{bt['n_test']:,} draws). At each draw every strategy sees only earlier draws and outputs a probability "
              "for each of the 50 main balls (summing to 5):", "",
              "- **Uniform:** 0.1 for every ball.",
              "- **Hot (all-time / last N):** frequency in the window, shrunk toward 0.1 by 10 pseudo-draws to avoid "
              "zero probabilities. Shrinkage only helps these strategies on log-loss; it does not change their top-5 picks.",
              "- **Cold / overdue:** a linear tilt by current-gap rank; the most overdue ball gets roughly twice the "
              "probability of the most recent one.",
              "- **Logistic regression:** features are appearance in each of the last 3 draws, frequency over the last "
              "10/20/50/100 draws and all-time, and log current gap. Trained on every (draw, ball) row from draw 100 "
              "up to the current point and refitted every 100 draws.", "",
              "Scores are mean per-ball log-loss and Brier score (lower is better) and mean main balls hit by each "
              f"strategy's top 5. The permutation p-value shuffles draw order {summary['n_permutations']:,} times and reruns the full backtest, "
              "including refitting the model, then asks how often a shuffled history scores at least as well. "
              "The HAC p-value tests whether the per-draw log-loss is below uniform, using Newey-West errors. "
              "The exact hits p-value uses the fact that under a fair draw any past-only pick hits "
              "hypergeometric(50,5,5) balls per draw, whatever the strategy.", ""]
    lines.append(md_table(tab, ["label", lambda r: f"{r['logloss']:.5f}", lambda r: f"{r['logloss_vs_uniform'] * 1000:+.3f}",
                                lambda r: f"{r['brier']:.5f}", lambda r: fp(r["p_beats_uniform_logloss"]),
                                lambda r: fp(r["p_perm_logloss"]), lambda r: fp(r["p_perm_brier"]),
                                lambda r: f"{r['mean_hits_top5']:.3f}", lambda r: fp(r["p_hits_exact"])],
                          ["Strategy", "Log-loss", "vs uniform (x1000)", "Brier", "p beats uniform (HAC)",
                           "p perm (log-loss)", "p perm (Brier)", "Top-5 hits/draw", "p hits (exact)"]))
    rh = bt["random_hits"]
    lines += ["", f"Random top-5 picks on the same draws: mean {rh.mean():.3f} hits per draw "
              f"(95% range {np.percentile(rh, 2.5):.3f}–{np.percentile(rh, 97.5):.3f}). The uniform row's top 5 "
              "is a random tie-break, so it is one such random pick.", "",
              f"![Backtest vs null]({FIG}/fig7_backtest_null.png)", "",
              f"![Cumulative log-loss]({FIG}/fig8_backtest_cumulative.png)", "",
              f"![Top-5 hits]({FIG}/fig9_backtest_hits.png)", ""]

    # ---- analysis 3
    low = res["low"]
    all_low = int((low["values"] == 5).sum())
    lines += ["## Analysis 3: avoiding shared jackpots", "",
              "Every combination has the same 1 in 139,838,160 chance. What differs is how many other players chose "
              "it, which decides how far a jackpot is split.", "",
              f"![Low numbers]({FIG}/fig10_low_numbers.png)", "",
              f"- 31 of the 50 main balls (62%) are possible days of the month. The draw itself does not care: winning "
              f"draws contain {low['mean']:.2f} balls of 31 or below on average against {low['expected']:.2f} expected "
              f"(chi-square p = {fp(low['p'])}), and all five were 31 or below in {all_low} draws "
              f"({all_low / len(low['values']) * 100:.1f}%, against {low['pmf'][5] * 100:.1f}% expected).",
              "- What is known about how people pick, from lottery studies that inferred popularity from published "
              "prize-tier winner counts on the UK National Lottery (Haigh 1997, J. R. Statist. Soc. A; and Ian Walker's "
              "analyses of UK Lotto winner data): people over-pick dates (1–31, and 1–12 especially), the number 7 and other "
              "'lucky' numbers, geometric patterns on the play slip (rows, columns, diagonals), arithmetic runs "
              "(1-2-3-4-5, 5-10-15-20-25), and the previous draw's winning numbers. Numbers above 31 and irregular, "
              "unpatterned combinations are under-picked. These findings come from 6-from-49 lotto games, not EuroMillions; the same "
              "biases are widely assumed for EuroMillions but I have not verified EuroMillions-specific figures.",
              "- **What public data can and can't tell you.** Operators do not publish how often each number or "
              "combination is chosen. They do publish, per draw, the number of winners in each prize tier (often per "
              "country) and sometimes sales. Comparing actual winners with the number expected from sales lets you "
              "estimate which drawn combinations were popular, which is how the studies above work. That data is not "
              "in this project's sources and the operator sites were not reachable from the build environment, so no "
              "popularity estimate is made here and none should be inferred from this report.",
              "- **Practical upshot.** Avoiding sharing does not change your odds of winning, only your expected "
              "payout if you win. Lucky Dip (random selection) already avoids human bias. If you choose your own "
              "numbers, include several above 31, avoid patterns and runs, and avoid last week's winning numbers.", ""]

    lines += ["## Reproducing", "", "```bash", "pip install -r requirements.txt",
              "python run_all.py            # fetch latest draws, rebuild, re-run everything, rewrite this report",
              "python run_all.py --offline  # re-run on the raw files already in data/raw/", "```", "",
              "All tables behind this report are in `outputs/tables/`.", ""]
    (C.ROOT / "report.md").write_text("\n".join(lines))
    (C.OUTPUTS / "summary.json").write_text(json.dumps(dict(
        data=summary,
        global_tests=tests[["test", "statistic", "p_used", "p_bh_family"]].to_dict(orient="records"),
        backtest=tab.drop(columns=["label"]).to_dict(orient="records"),
    ), indent=2, default=float))
