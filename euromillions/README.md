# EuroMillions: is there any signal in past draws?

A reproducible test of whether the full EuroMillions draw history (Feb 2004 onwards) carries any predictive signal. **Read `report.md` for the results.**

## Run

```bash
pip install -r requirements.txt
python run_all.py            # fetch the latest draws, rebuild, re-run every analysis, rewrite report.md
python run_all.py --offline  # same, using the raw files already in data/raw/
python run_all.py --quick    # 200 simulations / 50 permutations, for development only
```

Or `make all` / `make offline`. A full run takes about 15 minutes on 4 cores, nearly all of it the 1,000-permutation backtest. Seeds are fixed in `src/emstats/config.py`, so the same data always gives the same numbers.

## Layout

```
run_all.py                  one-command pipeline
src/emstats/
  config.py                 paths, sources, rule-change dates, seeds
  fetch.py                  downloads raw sources into data/raw/ and logs SHA-256s
  build.py                  reconciles sources, applies documented corrections, validates
  nullsim.py                fair-draw simulator and Monte Carlo p-values
  randomness.py             Analysis 1: frequency, gaps, serial dependence, pairs, sum, odd/even
  backtest.py               Analysis 2: walk-forward strategies + permutation null
  plots.py, report.py       charts and report.md
data/
  raw/                      source files exactly as downloaded
  manual_corrections.csv    every hand-applied fix, with its evidence
  processed/                draws.csv (validated), reconciliation_log.csv
outputs/
  figures/, tables/         everything report.md shows, plus full per-number/per-pair tables
```

## Adding new draws

The primary source updates itself on every draw day, so `python run_all.py` picks new draws up automatically. If the new data fails validation (a missing Tuesday or Friday, an off-schedule date, or a disagreement between sources), the build stops and says what is wrong. Fix it by adding a row with evidence to `data/manual_corrections.csv` (`redate`, `drop` or `add`). Each `redate` and `drop` row repeats the numbers it expects to find, so a correction cannot silently apply to the wrong row if the upstream file changes.
