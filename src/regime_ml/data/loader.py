"""Historical OHLCV loading and cleaning."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from regime_ml.config import RAW_DATA_DIR, ensure_output_dirs

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


class DataValidationError(ValueError):
    pass


def cache_path(symbol: str, period: str, interval: str) -> Path:
    safe = symbol.replace("/", "_").replace("^", "")
    return RAW_DATA_DIR / f"{safe}_{period}_{interval}.csv"


def clean_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df is None or df.empty:
        raise DataValidationError("Empty OHLCV frame.")

    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(c).strip().title() for c in data.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")

    data.index = pd.to_datetime(data.index)
    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_localize(None)

    initial = len(data)
    data = data[~data.index.duplicated(keep="first")].sort_index()
    data[list(REQUIRED_COLUMNS)] = data[list(REQUIRED_COLUMNS)].ffill()
    data = data.dropna(subset=list(REQUIRED_COLUMNS))

    bad = (data[["Open", "High", "Low", "Close"]] <= 0).any(axis=1) | (
        data["High"] < data["Low"]
    )
    data = data[~bad]
    if data.empty:
        raise DataValidationError("No valid rows after cleaning.")

    report = {
        "initial_rows": initial,
        "final_rows": len(data),
        "start": str(data.index.min().date()),
        "end": str(data.index.max().date()),
    }
    return data[list(REQUIRED_COLUMNS)], report


def load_ohlcv(
    symbol: str,
    period: str = "10y",
    interval: str = "1d",
    *,
    force_download: bool = False,
) -> tuple[pd.DataFrame, dict]:
    ensure_output_dirs()
    path = cache_path(symbol, period, interval)

    if path.exists() and not force_download:
        raw = pd.read_csv(path, index_col=0, parse_dates=True)
        data, report = clean_ohlcv(raw)
        report["source"] = "cache"
        return data, report

    downloaded = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if downloaded is None or downloaded.empty:
        raise DataValidationError(f"No data for {symbol}")

    data, report = clean_ohlcv(downloaded)
    data.to_csv(path)
    report["source"] = "download"
    return data, report


class Instrument:
    """Single-asset container with OHLCV and optional feature matrix."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.period: Optional[str] = None
        self.interval: Optional[str] = None
        self.clean_report: Optional[dict] = None

    def fetch(self, period: str, interval: str = "1d", *, force_download: bool = False) -> pd.DataFrame:
        self.period = period
        self.interval = interval
        self.data, self.clean_report = load_ohlcv(
            self.symbol, period=period, interval=interval, force_download=force_download
        )
        return self.data

    def require_data(self) -> pd.DataFrame:
        if self.data is None or self.data.empty:
            raise DataValidationError(f"No data for {self.symbol}. Call fetch() first.")
        return self.data
