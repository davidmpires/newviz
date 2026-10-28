"""One command to refresh everything: fetch -> reconcile -> analyse -> report.

    python run_all.py            # fetch latest draws first
    python run_all.py --offline  # use the raw files already in data/raw/
    python run_all.py --quick    # fewer simulations, for development only
    python run_all.py --report-only  # redraw charts and report.md from cached results
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emstats import backtest, build, fetch, plots, randomness, report  # noqa: E402
from emstats import config as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip downloading; use data/raw as is")
    ap.add_argument("--quick", action="store_true", help="200 simulations / 50 permutations (not for the report)")
    ap.add_argument("--report-only", action="store_true", help="re-render from outputs/results.pkl")
    args = ap.parse_args()
    cache = C.OUTPUTS / "results.pkl"
    if args.report_only:
        res, bt, summary, log, n = pickle.loads(cache.read_bytes())
        plots.make_all(res, bt, n)
        report.write(res, bt, summary, log)
        print("re-rendered report.md from cached results")
        return
    if args.quick:
        C.N_NULL_SIMS, C.N_PERMUTATIONS = 200, 50
    for d in (C.RAW, C.PROCESSED, C.FIGURES, C.TABLES):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("1/5 fetch")
    fetch.fetch_all(offline=args.offline)
    print("2/5 reconcile and validate")
    df, log, summary = build.build()
    print("3/5 analysis 1: randomness tests")
    res = randomness.run(df)
    print("4/5 analysis 2: walk-forward backtest")
    bt = backtest.run(df)
    summary.update(n_null_sims=C.N_NULL_SIMS, n_permutations=C.N_PERMUTATIONS)
    cache.write_bytes(pickle.dumps((res, bt, summary, log, len(df))))
    print("5/5 charts and report")
    plots.make_all(res, bt, len(df))
    report.write(res, bt, summary, log)
    print(f"done in {time.time() - t0:.0f}s -> report.md")


if __name__ == "__main__":
    main()
