"""Analysis 1: is the draw history consistent with a fair random draw?

Analytic tests are reported where they exist. Because the five balls in a
draw are sampled without replacement, several textbook tests are only
approximate, so every global statistic also gets a Monte Carlo p-value from
N_NULL_SIMS simulated fair histories of the same length.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from . import config as C
from .nullsim import indicator, mc_pvalue, simulate_indicator

MAINS = ["n1", "n2", "n3", "n4", "n5"]
STARS = ["s1", "s2"]
MAX_LAG = 10
# Per-number gap bins (each holds >= ~8% of the geometric(0.1) mass).
GAP_EDGES = [1, 2, 3, 4, 6, 8, 11, 15, 21, np.inf]
POOLED_GAP_MAX = 40


# ---------------------------------------------------------------- statistics

def freq_stat(X, k):
    """Chi-square for ball counts, corrected for sampling without replacement.

    With k of N drawn per draw, Cov(counts) = n p (1-p) N/(N-1) (I - J/N),
    so sum (O-E)^2 / (n p (1-p) N/(N-1)) ~ chi2(N-1) exactly in the limit.
    The plain Pearson statistic would be too small by a factor (1-p)N/(N-1).
    """
    n, N = X.shape
    p = k / N
    O = X.sum(axis=0)
    return ((O - n * p) ** 2).sum() / (n * p * (1 - p) * N / (N - 1))


def gaps_of(X):
    """Gaps (in draws) between successive appearances, per number."""
    return [np.diff(np.flatnonzero(X[:, i])) for i in range(X.shape[1])]


def geom_bin_probs(edges, p):
    lo, hi = np.array(edges[:-1]), np.array(edges[1:])
    cdf = lambda g: 1 - (1 - p) ** np.minimum(g - 1, 1e9)  # P(G <= g-1)
    return cdf(hi) - cdf(lo)


def pooled_gap_stat(gaps, p):
    allg = np.concatenate(gaps)
    edges = list(range(1, POOLED_GAP_MAX + 1)) + [np.inf]
    obs = np.histogram(allg, bins=edges)[0]
    exp = geom_bin_probs(edges, p) * len(allg)
    return ((obs - exp) ** 2 / exp).sum(), obs, exp, edges


def per_number_gap_stats(gaps, p):
    probs = geom_bin_probs(GAP_EDGES, p)
    out = []
    for g in gaps:
        obs = np.histogram(g, bins=GAP_EDGES)[0]
        exp = probs * len(g)
        out.append(((obs - exp) ** 2 / exp).sum())
    return np.array(out)


def overlap_means(X):
    Xf = X.astype(np.float32)
    return np.array([(Xf[:-k] * Xf[k:]).sum(axis=1).mean() for k in range(1, MAX_LAG + 1)])


def autocorr_z(X):
    """z-scores of each number's lag-k autocorrelation (k=1..MAX_LAG): (MAX_LAG, N)."""
    n = X.shape[0]
    Z = X - X.mean(axis=0)
    var = (Z ** 2).sum(axis=0)
    out = np.empty((MAX_LAG, X.shape[1]))
    for k in range(1, MAX_LAG + 1):
        r = (Z[:-k] * Z[k:]).sum(axis=0) / var
        out[k - 1] = r * np.sqrt(n)
    return out


def pair_counts(X):
    Xf = X.astype(np.float32)
    M = Xf.T @ Xf
    iu = np.triu_indices(X.shape[1], 1)
    return M[iu].astype(int), iu


def pair_dispersion(counts, n, p_pair):
    e = n * p_pair
    return ((counts - e) ** 2 / e).sum()


def exact_sum_distribution(N=C.MAIN_POOL, k=C.MAIN_K):
    """P(sum of k distinct balls from 1..N = s), by dynamic programming."""
    smax = sum(range(N - k + 1, N + 1))
    dp = np.zeros((k + 1, smax + 1))
    dp[0, 0] = 1
    for b in range(1, N + 1):
        for j in range(k, 0, -1):
            dp[j, b:] += dp[j - 1, : smax + 1 - b]
    counts = dp[k]
    return np.arange(smax + 1), counts / counts.sum()


def binned_chi2(obs_values, support, pmf, min_expected=5.0):
    """Chi-square GOF of discrete data against pmf, merging sparse bins."""
    n = len(obs_values)
    obs_full = pd.Series(obs_values).value_counts().reindex(support, fill_value=0).to_numpy()
    exp_full = pmf * n
    bins_o, bins_e, acc_o, acc_e = [], [], 0, 0.0
    for o, e in zip(obs_full, exp_full):
        acc_o += o
        acc_e += e
        if acc_e >= min_expected:
            bins_o.append(acc_o)
            bins_e.append(acc_e)
            acc_o, acc_e = 0, 0.0
    if acc_e > 0:
        bins_o[-1] += acc_o
        bins_e[-1] += acc_e
    bins_o, bins_e = np.array(bins_o), np.array(bins_e)
    chi2 = ((bins_o - bins_e) ** 2 / bins_e).sum()
    df = len(bins_o) - 1
    return chi2, df, stats.chi2.sf(chi2, df)


# ---------------------------------------------------------------- per-number tables

def per_number_table(X, k, label):
    n, N = X.shape
    p = k / N
    O = X.sum(axis=0)
    rows = []
    for i in range(N):
        bt = stats.binomtest(int(O[i]), n, p)
        ci = bt.proportion_ci(0.95, method="exact")
        rows.append(dict(set=label, number=i + 1, count=int(O[i]), expected=n * p,
                         ratio=O[i] / (n * p), ci_low=ci.low * n, ci_high=ci.high * n,
                         p_raw=bt.pvalue))
    t = pd.DataFrame(rows)
    t["p_bh"] = multipletests(t["p_raw"], method="fdr_bh")[1]
    t["p_bonferroni"] = multipletests(t["p_raw"], method="bonferroni")[1]
    t["significant_after_correction"] = (t["p_bh"] < C.ALPHA) | (t["p_bonferroni"] < C.ALPHA)
    return t


def pair_table(X):
    n, N = X.shape
    p_pair = C.MAIN_K * (C.MAIN_K - 1) / (N * (N - 1))
    counts, iu = pair_counts(X)
    pr = np.array([stats.binomtest(int(c), n, p_pair).pvalue for c in counts])
    t = pd.DataFrame(dict(a=iu[0] + 1, b=iu[1] + 1, count=counts, expected=n * p_pair, p_raw=pr))
    t["z"] = (t["count"] - t["expected"]) / np.sqrt(n * p_pair * (1 - p_pair))
    t["p_bh"] = multipletests(pr, method="fdr_bh")[1]
    t["p_bonferroni"] = multipletests(pr, method="bonferroni")[1]
    return t.sort_values("p_raw").reset_index(drop=True), p_pair


# ---------------------------------------------------------------- driver

def _mc_main(n, rng):
    """Null distributions of every global main-ball statistic."""
    p = C.MAIN_K / C.MAIN_POOL
    p_pair = C.MAIN_K * (C.MAIN_K - 1) / (C.MAIN_POOL * (C.MAIN_POOL - 1))
    out = {k: [] for k in ["freq", "gap_pooled", "gap_max_number", "overlap", "overlap_maxabs",
                           "autocorr_maxabs", "pair_disp", "pair_maxabs", "freq_maxabs"]}
    sd_overlap = np.sqrt(C.MAIN_K * p * (1 - p) * (C.MAIN_POOL - C.MAIN_K) / (C.MAIN_POOL - 1))
    for _ in range(C.N_NULL_SIMS):
        X = simulate_indicator(n, C.MAIN_POOL, C.MAIN_K, rng)
        out["freq"].append(freq_stat(X, C.MAIN_K))
        O = X.sum(axis=0)
        out["freq_maxabs"].append(np.abs((O - n * p) / np.sqrt(n * p * (1 - p))).max())
        g = gaps_of(X)
        out["gap_pooled"].append(pooled_gap_stat(g, p)[0])
        out["gap_max_number"].append(per_number_gap_stats(g, p).max())
        om = overlap_means(X)
        out["overlap"].append(om)
        zs = (om - 0.5) / (sd_overlap / np.sqrt(n - np.arange(1, MAX_LAG + 1)))
        out["overlap_maxabs"].append(np.abs(zs).max())
        out["autocorr_maxabs"].append(np.abs(autocorr_z(X)).max())
        pc, _ = pair_counts(X)
        out["pair_disp"].append(pair_dispersion(pc, n, p_pair))
        out["pair_maxabs"].append(np.abs((pc - n * p_pair) / np.sqrt(n * p_pair * (1 - p_pair))).max())
    return {k: np.array(v) for k, v in out.items()}


def run(df):
    rng = np.random.default_rng(C.SEED)
    res, tests = {}, []

    def add(test, statistic, df_, p_analytic, p_mc, note=""):
        tests.append(dict(test=test, statistic=statistic, df=df_, p_analytic=p_analytic,
                          p_monte_carlo=p_mc, note=note))

    # ---- main balls
    X = indicator(df[MAINS].to_numpy(), C.MAIN_POOL)
    n = len(X)
    p = C.MAIN_K / C.MAIN_POOL
    print(f"  simulating {C.N_NULL_SIMS} fair histories of {n} draws ...")
    null = _mc_main(n, rng)
    res["null"] = null

    T = freq_stat(X, C.MAIN_K)
    add("Main ball frequencies (chi-square, 49 df)", T, 49, stats.chi2.sf(T, 49), mc_pvalue(T, null["freq"]))
    main_tab = per_number_table(X, C.MAIN_K, "main")
    res["main_table"] = main_tab
    maxz = np.abs((main_tab["count"] - main_tab["expected"]) / np.sqrt(n * p * (1 - p))).max()
    add("Main ball: most extreme single number (max |z|)", maxz, None, None,
        mc_pvalue(maxz, null["freq_maxabs"]),
        f"{int((main_tab['p_raw'] < 0.05).sum())} of 50 nominally significant at 5%; "
        f"{int(main_tab['significant_after_correction'].sum())} after BH/Bonferroni")

    # gaps
    g = gaps_of(X)
    G, gobs, gexp, gedges = pooled_gap_stat(g, p)
    add(f"Gap distribution, pooled vs geometric(0.1) (chi-square, {len(gobs) - 1} bins-1)", G,
        len(gobs) - 1, stats.chi2.sf(G, len(gobs) - 1), mc_pvalue(G, null["gap_pooled"]),
        "analytic p approximate: gaps are not independent across numbers")
    per_gap = per_number_gap_stats(g, p)
    gp = stats.chi2.sf(per_gap, len(GAP_EDGES) - 2)
    gap_tab = pd.DataFrame(dict(number=np.arange(1, 51), n_gaps=[len(x) for x in g],
                                mean_gap=[x.mean() for x in g], max_gap=[x.max() for x in g],
                                chi2=per_gap, p_raw=gp,
                                p_bh=multipletests(gp, method="fdr_bh")[1],
                                p_bonferroni=multipletests(gp, method="bonferroni")[1]))
    res["gap_table"], res["gap_pooled"] = gap_tab, (gobs, gexp, gedges)
    add("Gap distribution, worst single number (max chi-square)", per_gap.max(), None, None,
        mc_pvalue(per_gap.max(), null["gap_max_number"]),
        f"{int((gap_tab['p_bh'] < C.ALPHA).sum())} numbers significant after BH")

    # serial dependence
    om = overlap_means(X)
    sd_ov = np.sqrt(C.MAIN_K * p * (1 - p) * (C.MAIN_POOL - C.MAIN_K) / (C.MAIN_POOL - 1))
    lag_n = n - np.arange(1, MAX_LAG + 1)
    oz = (om - 0.5) / (sd_ov / np.sqrt(lag_n))
    lo_b, hi_b = np.percentile(null["overlap"], [2.5, 97.5], axis=0)
    hyp = stats.hypergeom(C.MAIN_POOL, C.MAIN_K, C.MAIN_K)
    lag_rows = []
    Xf = X.astype(np.float32)
    for k in range(1, MAX_LAG + 1):
        ov = (Xf[:-k] * Xf[k:]).sum(axis=1).astype(int)
        chi2, dfk, pk = binned_chi2(ov, np.arange(0, 6), hyp.pmf(np.arange(0, 6)))
        lag_rows.append(dict(lag=k, mean_overlap=om[k - 1], expected=0.5, z=oz[k - 1],
                             p_raw=2 * stats.norm.sf(abs(oz[k - 1])), dist_chi2=chi2, dist_df=dfk, dist_p=pk,
                             null_low=lo_b[k - 1], null_high=hi_b[k - 1]))
    lag_tab = pd.DataFrame(lag_rows)
    lag_tab["p_bh"] = multipletests(lag_tab["p_raw"], method="fdr_bh")[1]
    res["lag_table"] = lag_tab
    add("Serial overlap, draws t and t+k, k=1..10 (max |z|)", np.abs(oz).max(), None, None,
        mc_pvalue(np.abs(oz).max(), null["overlap_maxabs"]),
        f"lag-1 mean overlap {om[0]:.3f} vs 0.500 expected")
    az = autocorr_z(X)
    ap = 2 * stats.norm.sf(np.abs(az)).ravel()
    res["autocorr_bh_sig"] = int((multipletests(ap, method="fdr_bh")[1] < C.ALPHA).sum())
    add("Per-number autocorrelation, 50 numbers x 10 lags (max |z|)", np.abs(az).max(), None, None,
        mc_pvalue(np.abs(az).max(), null["autocorr_maxabs"]),
        f"{int((ap < 0.05).sum())} of 500 nominally significant; {res['autocorr_bh_sig']} after BH")

    # pairs
    pair_tab, p_pair = pair_table(X)
    res["pair_table"], res["p_pair"] = pair_tab, p_pair
    D = pair_dispersion(pair_tab["count"].to_numpy(), n, p_pair)
    add("Pair co-occurrence dispersion, 1,225 pairs", D, None, None, mc_pvalue(D, null["pair_disp"]),
        f"{int((pair_tab['p_raw'] < 0.05).sum())} pairs nominally significant (~61 expected by chance); "
        f"{int((pair_tab['p_bh'] < C.ALPHA).sum())} after BH")
    pmax = np.abs(pair_tab["z"]).max()
    add("Pair co-occurrence, most extreme pair (max |z|)", pmax, None, None, mc_pvalue(pmax, null["pair_maxabs"]))

    # sum and odd/even
    sums = df[MAINS].sum(axis=1).to_numpy()
    support, pmf = exact_sum_distribution()
    chi2, dfs, ps = binned_chi2(sums, support, pmf)
    mu = (support * pmf).sum()
    sd = np.sqrt(((support - mu) ** 2 * pmf).sum())
    add("Sum of main balls vs exact distribution (chi-square)", chi2, dfs, ps, None,
        f"observed mean {sums.mean():.2f} vs {mu:.2f}; sd {sums.std(ddof=1):.2f} vs {sd:.2f}")
    res["sum"] = (sums, support, pmf)
    null_sd = rng.choice(support, p=pmf, size=(C.N_NULL_SIMS, n)).std(axis=1, ddof=1)
    obs_sd = sums.std(ddof=1)
    p_sd = min(1.0, 2 * min(mc_pvalue(obs_sd, null_sd), mc_pvalue(obs_sd, null_sd, greater=False)))
    add("Sum of main balls: standard deviation vs exact (two-sided)", obs_sd, None, None, p_sd,
        f"observed sd {obs_sd:.2f} vs {sd:.2f} expected")
    odd = (df[MAINS] % 2 == 1).sum(axis=1).to_numpy()
    hodd = stats.hypergeom(C.MAIN_POOL, 25, C.MAIN_K).pmf(np.arange(6))
    chi2, dfo, po = binned_chi2(odd, np.arange(6), hodd)
    add("Odd-ball count per draw vs hypergeometric (chi-square)", chi2, dfo, po, None)
    res["odd"] = (odd, hodd)

    # ---- Lucky Stars, within each era only
    star_tabs = []
    for label, _, pool in C.ERAS:
        sub = df[df["era"] == label]
        Xs = indicator(sub[STARS].to_numpy(), pool)
        ns = len(Xs)
        Ts = freq_stat(Xs, C.STAR_K)
        nullT = np.array([freq_stat(simulate_indicator(ns, pool, C.STAR_K, rng), C.STAR_K)
                          for _ in range(C.N_NULL_SIMS)])
        add(f"Lucky Stars era {label} ({ns} draws) frequencies (chi-square, {pool - 1} df)", Ts, pool - 1,
            stats.chi2.sf(Ts, pool - 1), mc_pvalue(Ts, nullT))
        t = per_number_table(Xs, C.STAR_K, f"stars {label}")
        t["era"], t["n_draws"] = label, ns
        star_tabs.append(t)
    res["star_table"] = pd.concat(star_tabs, ignore_index=True)

    tests = pd.DataFrame(tests)
    # Benjamini-Hochberg across the family of global tests (MC p where available).
    pp = tests["p_monte_carlo"].fillna(tests["p_analytic"])
    tests["p_used"] = pp
    tests["p_bh_family"] = multipletests(pp, method="fdr_bh")[1]
    res["tests"] = tests

    # low numbers (for Analysis 3)
    low = (df[MAINS] <= 31).sum(axis=1).to_numpy()
    hlow = stats.hypergeom(C.MAIN_POOL, 31, C.MAIN_K).pmf(np.arange(6))
    chi2, dfl, pl = binned_chi2(low, np.arange(6), hlow)
    res["low"] = dict(values=low, pmf=hlow, chi2=chi2, df=dfl, p=pl, mean=low.mean(), expected=5 * 31 / 50)

    # save tables
    tests.to_csv(C.TABLES / "analysis1_global_tests.csv", index=False)
    main_tab.to_csv(C.TABLES / "analysis1_main_per_number.csv", index=False)
    res["star_table"].to_csv(C.TABLES / "analysis1_stars_per_number.csv", index=False)
    gap_tab.to_csv(C.TABLES / "analysis1_gaps_per_number.csv", index=False)
    lag_tab.to_csv(C.TABLES / "analysis1_serial_overlap.csv", index=False)
    pair_tab.to_csv(C.TABLES / "analysis1_pairs.csv", index=False)
    return res
