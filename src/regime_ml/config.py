"""Project paths and research defaults."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# Thesis constants (single place to change them)
MAX_HOLD_DAYS = 60
BASE_COST = 0.0005
SLIPPAGE_ALPHA = 0.9
REGIME_CONF_THRESHOLD = 0.75
MAX_SHARPE_CAP = 3.0
TRAIN_RATIO = 0.70
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0

REGIME_FEATURES = [
    "Trend_Normalized",
    "Volatility_Stability",
    "Momentum_Normalized",
    "Vol_Change",
]

STRATEGY_NAMES = [
    "MovingAvg",
    "Momentum",
    "MeanRev",
    "Breakout",
    "TrendFollow",
]

DEFAULT_WINDOWS = [
    ("2017-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2019-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2020-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("2021-01-01", "2025-12-31", "2026-01-01", "2026-06-01"),
]

ASSET_UNIVERSES = {
    "mega_tech": ["AAPL", "TSLA", "NVDA", "GOOGL", "META", "AMZN", "MSFT"],
    "core_equity": ["SPY"],
    "crypto": ["BTC-USD"],
}


def ensure_output_dirs() -> None:
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR):
        path.mkdir(parents=True, exist_ok=True)
