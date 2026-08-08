"""Transaction cost helpers."""

from __future__ import annotations

import pandas as pd

from regime_ml.config import SLIPPAGE_ALPHA


def compute_slippage(close: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Slippage_t = α * σ²_20day

    σ² is the lookback rolling variance of daily returns.
    """
    variance = close.pct_change().rolling(lookback).var()
    variance = variance.fillna(close.pct_change().var())
    return SLIPPAGE_ALPHA * variance
