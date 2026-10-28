"""Analysis 2: walk-forward backtest of number-picking strategies (main balls).

For every draw t >= BURN_IN each strategy sees only draws 0..t-1 and outputs
a probability for each of the 50 numbers (summing to 5). Scores are compared
with the uniform baseline and with a permutation null built by shuffling the
draw order, which keeps every draw intact but destroys any time structure.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .nullsim import indicator, mc_pvalue

MAINS = ["n1", "n2", "n3", "n4", "n5"]
P0 = C.MAIN_K / C.MAIN_POOL
WINDOWS = (20, 50, 100)
SMOOTH = 10          # pseudo-draws of uniform prior added to every frequency estimate
ML_REFIT_EVERY = 100
ML_MIN_HISTORY = 100
EPS = 1e-6

STRATEGIES = ["uniform", "hot_all_time"] + [f"hot_last_{w}" for w in WINDOWS] + ["cold_overdue", "logistic_regression"]
LABELS = {
    "uniform": "Uniform baseline",
    "hot_all_time": "Hot: all-time frequency",
    **{f"hot_last_{w}": f"Hot: last {w} draws" for w in WINDOWS},
    "cold_overdue": "Cold / overdue (longest gap)",
    "logistic_regression": "Logistic regression (lags, windows, gap)",
}


def features(X):
    """State before each draw t. All arrays are (n, 50) and use draws < t only."""
    n, N = X.shape
    cs = np.vstack([np.zeros((1, N)), np.cumsum(X, axis=0)])   # cs[t] = sum of X[:t]
    t = np.arange(n)[:, None]
    f = {"cum": cs[:n], "t": np.broadcast_to(t, (n, N)).astype(float)}
    for w in (3, 10) + WINDOWS:
        f[f"win{w}"] = cs[:n] - cs[np.maximum(np.arange(n) - w, 0)]
        f[f"len{w}"] = np.minimum(t, w).astype(float)
    last = np.full(N, -1)
    gap = np.empty((n, N))
    for i in range(n):
        gap[i] = i - last           # draws since last seen (i+1 if never)
        last = np.where(X[i] == 1, i, last)
    f["gap"] = gap
    lag = np.zeros((3, n, N))
    for k in range(1, 4):
        lag[k - 1, k:] = X[:-k]
    f["lag"] = lag
    return f


def _freq_prob(count, length):
    return (count + P0 * SMOOTH) / (length + SMOOTH)


def _ml_design(f):
    with np.errstate(divide="ignore", invalid="ignore"):
        cols = [f["lag"][0], f["lag"][1], f["lag"][2]]
        cols += [np.where(f[f"len{w}"] > 0, f[f"win{w}"] / f[f"len{w}"], P0) for w in (10,) + WINDOWS]
        cols += [np.where(f["t"] > 0, f["cum"] / np.maximum(f["t"], 1), P0), np.log(f["gap"])]
    return np.stack(cols, axis=-1)  # (n, 50, n_features)


def predict_all(X):
    """Probability matrices (n_test, 50) for every strategy."""
    n = len(X)
    f = features(X)
    idx = np.arange(C.BURN_IN, n)
    P = {"uniform": np.full((len(idx), C.MAIN_POOL), P0)}
    P["hot_all_time"] = _freq_prob(f["cum"][idx], f["t"][idx])
    for w in WINDOWS:
        P[f"hot_last_{w}"] = _freq_prob(f[f"win{w}"][idx], f[f"len{w}"][idx])
    # Overdue: a mild linear tilt toward numbers with the longest current gap.
    ranks = stats.rankdata(f["gap"][idx], axis=1)
    w = 1 + ranks / C.MAIN_POOL
    P["cold_overdue"] = C.MAIN_K * w / w.sum(axis=1, keepdims=True)
    # Logistic regression, refitted every ML_REFIT_EVERY draws on all earlier draws.
    D, out = _ml_design(f), np.empty((len(idx), C.MAIN_POOL))
    for start in range(C.BURN_IN, n, ML_REFIT_EVERY):
        tr = slice(ML_MIN_HISTORY, start)
        Xtr = D[tr].reshape(-1, D.shape[-1])
        ytr = X[tr].reshape(-1)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
        model.fit(Xtr, ytr)
        stop = min(start + ML_REFIT_EVERY, n)
        pr = model.predict_proba(D[start:stop].reshape(-1, D.shape[-1]))[:, 1].reshape(stop - start, -1)
        out[start - C.BURN_IN: stop - C.BURN_IN] = pr
    P["logistic_regression"] = C.MAIN_K * out / out.sum(axis=1, keepdims=True)
    return P


def score(P, Y, jitter):
    """Per-draw log-loss, Brier and top-5 hits. jitter breaks ties reproducibly."""
    Pc = np.clip(P, EPS, 1 - EPS)
    ll = -(Y * np.log(Pc) + (1 - Y) * np.log(1 - Pc)).mean(axis=1)
    br = ((P - Y) ** 2).mean(axis=1)
    top = np.argpartition(-(P + jitter), C.MAIN_K, axis=1)[:, : C.MAIN_K]
    hits = np.take_along_axis(Y, top, axis=1).sum(axis=1)
    return ll, br, hits


def evaluate(X, seed):
    rng = np.random.default_rng(seed)
    Y = X[C.BURN_IN:].astype(float)
    jitter = rng.random(Y.shape) * 1e-9
    P = predict_all(X)
    return {s: score(P[s], Y, jitter) for s in STRATEGIES}


def _perm_summary(X, seed):
    rng = np.random.default_rng(seed)
    Xp = X[rng.permutation(len(X))]
    r = evaluate(Xp, seed)
    return {s: (v[0].mean(), v[1].mean(), v[2].mean()) for s, v in r.items()}


def hac_test(diff, maxlags=10):
    """Mean loss differential with Newey-West SE; one-sided p for 'strategy better'."""
    fit = sm.OLS(diff, np.ones_like(diff)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    t = fit.params[0] / fit.bse[0]
    return fit.params[0], t, stats.norm.cdf(t)


def total_hits_null_pvalue(total, n_draws):
    """Under a fair draw, top-5 hits of ANY past-only strategy are iid hypergeom(50,5,5)."""
    pmf = stats.hypergeom(C.MAIN_POOL, C.MAIN_K, C.MAIN_K).pmf(np.arange(6))
    dist = np.array([1.0])
    for _ in range(n_draws):
        dist = np.convolve(dist, pmf)
    return dist[int(total):].sum()


def run(df):
    X = indicator(df[MAINS].to_numpy(), C.MAIN_POOL)
    n_test = len(X) - C.BURN_IN
    obs = evaluate(X, C.SEED)

    # Random top-5 picks on the same draws, for the hits comparison.
    rng = np.random.default_rng(C.SEED + 1)
    rand_hits = np.array([
        X[C.BURN_IN:][np.arange(n_test)[:, None], np.argsort(rng.random((n_test, 50)), axis=1)[:, :5]].sum(axis=1).mean()
        for _ in range(1000)])

    print(f"  backtest: {n_test} test draws; running {C.N_PERMUTATIONS} draw-order permutations ...")
    seeds = np.random.SeedSequence(C.SEED).generate_state(C.N_PERMUTATIONS)
    perms = Parallel(n_jobs=-1)(delayed(_perm_summary)(X, int(s)) for s in seeds)

    uni_ll = obs["uniform"][0]
    uni_br = obs["uniform"][1]
    rows, null = [], {}
    for s in STRATEGIES:
        ll, br, hits = obs[s]
        nll = np.array([p[s][0] for p in perms])
        nbr = np.array([p[s][1] for p in perms])
        nh = np.array([p[s][2] for p in perms])
        null[s] = dict(logloss=nll, brier=nbr, hits=nh)
        d_ll, t_ll, p_ll = hac_test(ll - uni_ll) if s != "uniform" else (0.0, np.nan, np.nan)
        rows.append(dict(
            strategy=s, label=LABELS[s],
            logloss=ll.mean(), logloss_vs_uniform=ll.mean() - uni_ll.mean(),
            brier=br.mean(), brier_vs_uniform=br.mean() - uni_br.mean(),
            hac_t_vs_uniform=t_ll, p_beats_uniform_logloss=p_ll,
            p_perm_logloss=mc_pvalue(ll.mean(), nll, greater=False) if s != "uniform" else np.nan,
            p_perm_brier=mc_pvalue(br.mean(), nbr, greater=False) if s != "uniform" else np.nan,
            mean_hits_top5=hits.mean(), total_hits=int(hits.sum()),
            p_hits_exact=total_hits_null_pvalue(hits.sum(), n_test),
            p_perm_hits=mc_pvalue(hits.mean(), nh),
        ))
    tab = pd.DataFrame(rows)
    tab.to_csv(C.TABLES / "analysis2_backtest.csv", index=False)
    per_draw = pd.DataFrame({"date": df["date"].iloc[C.BURN_IN:].to_numpy(),
                             **{f"{s}_logloss": obs[s][0] for s in STRATEGIES},
                             **{f"{s}_hits": obs[s][2] for s in STRATEGIES}})
    per_draw.to_csv(C.TABLES / "analysis2_per_draw_scores.csv", index=False)
    return dict(table=tab, null=null, obs=obs, per_draw=per_draw, n_test=n_test,
                random_hits=rand_hits, test_start=str(df["date"].iloc[C.BURN_IN].date()))
