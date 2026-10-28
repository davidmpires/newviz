"""Project-wide constants: paths, sources, game rules and seeds."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MANUAL_CORRECTIONS = DATA / "manual_corrections.csv"

# Primary source: full history, updated automatically on every draw day.
# Upstream of this file is win2day.at (Österreichische Lotterien, an official
# EuroMillions operator).
PRIMARY_URL = (
    "https://raw.githubusercontent.com/daowa89/lottery-archive/main/"
    "eu/euromillions/results.csv"
)
PRIMARY_FILE = RAW / "lottery_archive_euromillions.csv"

# Secondary source for cross-checking: the result files bundled in the
# pyeuromil package (frozen; covers 10 May 2011 to 3 Dec 2021).
SECONDARY_PACKAGE = "pyeuromil"
SECONDARY_VERSION = "0.2.20211203"
SECONDARY_DIR = RAW / "pyeuromil"

FETCH_LOG = RAW / "fetch_log.json"

# Game rules. Main balls have always been 5 from 1-50.
MAIN_POOL, MAIN_K = 50, 5
STAR_K = 2
# Lucky Star eras: (label, first draw date, pool size).
ERAS = [
    ("1-9", pd.Timestamp("2004-02-13"), 9),
    ("1-11", pd.Timestamp("2011-05-10"), 11),
    ("1-12", pd.Timestamp("2016-09-27"), 12),
]
FIRST_DRAW = ERAS[0][1]
# Fridays only until 6 May 2011; Tuesdays and Fridays from 10 May 2011.
TUESDAY_START = pd.Timestamp("2011-05-10")

# Fixed seeds so every run gives identical numbers for identical data.
SEED = 20040213
N_NULL_SIMS = 2000        # Monte Carlo histories for Analysis 1
N_PERMUTATIONS = 1000     # draw-order shuffles for Analysis 2
BURN_IN = 200             # first draw index scored in the backtest
ALPHA = 0.05
