"""Static charts for report.md (matplotlib, PNG)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import stats  # noqa: E402

from . import config as C  # noqa: E402
from .backtest import LABELS, STRATEGIES  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
BAND = "#dcdbd5"
BAND2 = "#eeede9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OBS, REF, FLAG = SERIES[0], SERIES[1], "#e34948"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "figure.dpi": 110,
})


def _save(fig, name):
    path = C.FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return path


def _freq_panel(ax, tab, n, k, pool):
    p = k / pool
    x = tab["number"].to_numpy()
    e = n * p
    sd = np.sqrt(n * p * (1 - p))
    z95 = stats.norm.ppf(0.975)
    zb = stats.norm.ppf(1 - 0.025 / pool)
    ax.fill_between([0.4, pool + 0.6], e - zb * sd, e + zb * sd, color=BAND2, lw=0, label=f"Bonferroni 95% band ({pool} tests)")
    ax.fill_between([0.4, pool + 0.6], e - z95 * sd, e + z95 * sd, color=BAND, lw=0, label="Pointwise 95% band")
    colors = [FLAG if s else OBS for s in tab["significant_after_correction"]]
    ax.bar(x, tab["count"], width=0.62, color=colors, zorder=3)
    ax.axhline(e, color=INK, lw=1, ls="--", zorder=4, label=f"Expected ({e:.1f})")
    lo = min(tab["count"].min(), e - zb * sd)
    hi = max(tab["count"].max(), e + zb * sd)
    pad = (hi - lo) * 0.25
    ax.set_ylim(max(0, lo - pad), hi + pad * 0.6)
    ax.set_xlim(0.4, pool + 0.6)
    ax.grid(axis="x", visible=False)


def main_frequency(res, n):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    _freq_panel(ax, res["main_table"], n, C.MAIN_K, C.MAIN_POOL)
    ax.set_xticks(range(1, 51, 1))
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("Main ball")
    ax.set_ylabel("Times drawn")
    ax.set_title(f"Main ball frequencies, {n:,} draws, against the fair-draw expectation")
    ax.legend(loc="upper left", ncol=3)
    return _save(fig, "fig1_main_frequency")


def star_frequency(res):
    st = res["star_table"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [9, 11, 12]})
    for ax, (label, _, pool) in zip(axes, C.ERAS):
        t = st[st["era"] == label]
        _freq_panel(ax, t, int(t["n_draws"].iloc[0]), C.STAR_K, pool)
        ax.set_xticks(range(1, pool + 1))
        ax.set_title(f"Era {label}: {int(t['n_draws'].iloc[0])} draws", fontsize=10)
        ax.set_xlabel("Lucky Star")
    axes[0].set_ylabel("Times drawn")
    axes[0].legend(loc="upper left", fontsize=6.5)
    fig.suptitle("Lucky Star frequencies within each era", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig2_star_frequency")


def gaps(res):
    obs, exp, edges = res["gap_pooled"]
    tot = obs.sum()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.array(edges[:-2])  # the open-ended tail bin is tested but not drawn
    a1.bar(x, obs[:-1] / tot, width=0.7, color=OBS, label="Observed (all numbers pooled)")
    a1.plot(x, exp[:-1] / tot, color=INK, lw=1.4, marker="o", ms=2.5, label="Geometric(p = 0.1)")
    a1.set_yscale("log")
    a1.set_xlabel(f"Draws between appearances ({edges[-2]}+ tail tested, not drawn)")
    a1.set_ylabel("Share of gaps (log scale)")
    a1.set_title("Gap distribution vs geometric")
    a1.legend()
    gt = res["gap_table"]
    se = np.sqrt(0.9) / 0.1 / np.sqrt(gt["n_gaps"].to_numpy())
    a2.fill_between(gt["number"], 10 - 1.96 * se, 10 + 1.96 * se, color=BAND, lw=0, step="mid", label="Pointwise 95% band")
    a2.axhline(10, color=INK, lw=1, ls="--", label="Expected mean gap (10)")
    a2.scatter(gt["number"], gt["mean_gap"], s=14, color=OBS, zorder=3, label="Observed mean gap")
    a2.set_xlabel("Main ball")
    a2.set_ylabel("Mean gap (draws)")
    a2.set_title("Mean gap per number")
    a2.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return _save(fig, "fig3_gaps")


def serial(res):
    lt = res["lag_table"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.fill_between(lt["lag"], lt["null_low"], lt["null_high"], color=BAND, lw=0, label="95% range in fair simulations")
    ax.axhline(0.5, color=INK, lw=1, ls="--", label="Expected (0.5)")
    ax.plot(lt["lag"], lt["mean_overlap"], color=OBS, lw=2, marker="o", ms=5, label="Observed")
    ax.set_xticks(lt["lag"])
    ax.set_xlabel("Lag k (draws)")
    ax.set_ylabel("Mean shared main balls")
    ax.set_title("Main balls shared between draw t and draw t+k")
    ax.legend(loc="lower right")
    return _save(fig, "fig4_serial_overlap")


def pairs(res, n):
    pt = res["pair_table"]
    p = res["p_pair"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1]})
    c = pt["count"].to_numpy()
    ks = np.arange(c.min() - 2, c.max() + 3)
    obs = np.bincount(c, minlength=ks.max() + 1)[ks]
    a1.bar(ks, obs, width=0.7, color=OBS, label="Observed pairs")
    a1.plot(ks, stats.binom.pmf(ks, n, p) * len(c), color=INK, lw=1.4, marker="o", ms=2.5,
            label=f"Binomial(n={n}, p={p:.4f})")
    a1.set_xlabel("Times a pair was drawn together")
    a1.set_ylabel("Number of pairs")
    a1.set_title("Pair co-occurrence counts (1,225 pairs)")
    a1.legend(loc="upper left")
    M = np.full((50, 50), np.nan)
    M[pt["a"] - 1, pt["b"] - 1] = pt["z"]
    M[pt["b"] - 1, pt["a"] - 1] = pt["z"]
    im = a2.imshow(M, cmap=DIVERGING, vmin=-4, vmax=4, origin="lower", extent=[0.5, 50.5, 0.5, 50.5])
    a2.grid(False)
    a2.set_title("Pair z-scores (observed minus expected)")
    a2.set_xlabel("Main ball")
    a2.set_ylabel("Main ball")
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.03)
    cb.set_label("z", color=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return _save(fig, "fig5_pairs")


def sum_oddeven(res):
    sums, support, pmf = res["sum"]
    odd, hodd = res["odd"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    edges = np.arange(10, 251, 10)
    obs = np.histogram(sums, bins=edges)[0]
    exp = np.array([pmf[(support >= lo) & (support < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])]) * len(sums)
    mids = edges[:-1] + 5
    a1.bar(mids, obs, width=8, color=OBS, label="Observed")
    a1.plot(mids, exp, color=INK, lw=1.4, marker="o", ms=2.5, label="Exact distribution")
    a1.set_xlabel("Sum of the five main balls (bins of 10)")
    a1.set_ylabel("Draws")
    a1.set_title("Sum of main balls")
    a1.legend()
    k = np.arange(6)
    oc = np.bincount(odd, minlength=6)
    a2.bar(k - 0.18, oc, width=0.34, color=OBS, label="Observed")
    a2.bar(k + 0.18, hodd * len(odd), width=0.34, color=BAND, label="Hypergeometric")
    a2.set_xticks(k)
    a2.set_xlabel("Odd main balls in the draw")
    a2.set_ylabel("Draws")
    a2.set_title("Odd/even split")
    a2.grid(axis="x", visible=False)
    a2.legend()
    fig.tight_layout()
    return _save(fig, "fig6_sum_oddeven")


def backtest_null(bt):
    tab = bt["table"].set_index("strategy")
    uni = tab.loc["uniform", "logloss"]
    strats = [s for s in STRATEGIES if s != "uniform"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, s in zip(axes.ravel(), strats):
        null = (bt["null"][s]["logloss"] - uni) * 1000
        obs = (tab.loc[s, "logloss"] - uni) * 1000
        ax.hist(null, bins=40, color=BAND, edgecolor=SURFACE, linewidth=0.5, label="Shuffled draw order")
        ax.axvline(0, color=REF, lw=1.6, ls="--", label="Uniform baseline")
        ax.axvline(obs, color=OBS, lw=2.2, label="Actual order")
        ax.set_title(f"{LABELS[s]}\npermutation p = {tab.loc[s, 'p_perm_logloss']:.2f}", fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", left=False, labelleft=False)
    for ax in axes[1]:
        ax.set_xlabel("Mean log-loss minus uniform (x1000)\nleft of the dashed line = better than uniform")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"Backtest log-loss vs null distribution ({len(bt['null']['uniform']['logloss']):,} draw-order shuffles)", x=0.01,
                 ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8)
    return _save(fig, "fig7_backtest_null")


def backtest_cumulative(bt):
    pdw = bt["per_draw"]
    uni = pdw["uniform_logloss"].to_numpy()
    strats = [s for s in STRATEGIES if s != "uniform"]
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, s in enumerate(strats):
        y = np.cumsum((pdw[f"{s}_logloss"].to_numpy() - uni) * C.MAIN_POOL)
        ax.plot(pdw["date"], y, color=SERIES[i], lw=2, label=LABELS[s])
    ax.axhline(0, color=INK, lw=1, ls="--")
    ax.set_ylabel("Cumulative excess log-loss (nats)\nabove zero = worse than uniform")
    ax.set_title("Cumulative log-loss relative to the uniform baseline, walk-forward")
    ax.legend(loc="upper left", ncol=2)
    return _save(fig, "fig8_backtest_cumulative")


def backtest_hits(bt):
    tab = bt["table"]
    rh = bt["random_hits"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    lo, hi = np.percentile(rh, [2.5, 97.5])
    ax.axvspan(lo, hi, color=BAND, lw=0, label="95% range, random top-5 picks")
    ax.axvline(0.5, color=INK, lw=1, ls="--", label="Expected (0.5)")
    y = np.arange(len(tab))[::-1]
    ax.scatter(tab["mean_hits_top5"], y, s=40, color=OBS, zorder=3)
    for yy, v in zip(y, tab["mean_hits_top5"]):
        ax.text(v + 0.004, yy, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(tab["label"])
    ax.set_xlabel("Mean main balls hit per draw by the strategy's top 5")
    ax.set_title("Top-5 hits per draw")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7)
    return _save(fig, "fig9_backtest_hits")


def low_numbers(res):
    low = res["low"]
    k = np.arange(6)
    oc = np.bincount(low["values"], minlength=6)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(k - 0.18, oc, width=0.34, color=OBS, label="Observed")
    ax.bar(k + 0.18, low["pmf"] * len(low["values"]), width=0.34, color=BAND, label="Hypergeometric")
    ax.set_xticks(k)
    ax.set_xlabel("Main balls of 31 or below (\"birthday\" numbers) in the draw")
    ax.set_ylabel("Draws")
    ax.set_title("How many winning main balls are 1-31")
    ax.grid(axis="x", visible=False)
    ax.legend()
    return _save(fig, "fig10_low_numbers")


def make_all(res, bt, n):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    return [main_frequency(res, n), star_frequency(res), gaps(res), serial(res), pairs(res, n),
            sum_oddeven(res), backtest_null(bt), backtest_cumulative(bt), backtest_hits(bt), low_numbers(res)]
