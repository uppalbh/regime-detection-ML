"""Regime features for the GMM detector.

These four features are the thesis feature set. They are built only from
information available at each timestamp.
"""

from __future__ import annotations

import pandas as pd

from regime_ml.config import REGIME_FEATURES
from regime_ml.features.indicators import add_base_indicators


def build_regime_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer normalized trend / momentum / volatility features for GMM input.

    Returns a DataFrame indexed like the input, restricted to business days,
    with NaN warmup rows dropped.
    """
    df = add_base_indicators(ohlcv)

    base_trend = (df["Close"] - df["EMA_20"]) / df["EMA_20"]
    vol_regime = df["Rolling_STD_20"] / df["Rolling_STD_20"].rolling(50, min_periods=50).mean()
    base_momentum = df["RSI_14"]

    df["Trend_Normalized"] = base_trend / vol_regime.replace(0.0, pd.NA)
    df["Momentum_Normalized"] = base_momentum / vol_regime.replace(0.0, pd.NA)
    df["Volatility_Stability"] = vol_regime.rolling(20, min_periods=20).std()
    df["Vol_Change"] = (df["ATR_14"] / df["ATR_14"].shift(5)) - 1.0

    # Keep OHLCV + engineered columns needed downstream
    keep = list(ohlcv.columns) + list(REGIME_FEATURES) + [
        "EMA_20",
        "EMA_50",
        "SMA_20",
        "RSI_14",
        "Rolling_STD_20",
        "ATR_14",
        "Rolling_Max_50",
        "Rolling_Min_50",
        "MACD_Histogram",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    # Business days only (matches thesis walk-forward filter)
    out = out[out.index.dayofweek < 5]
    out = out.dropna(subset=list(REGIME_FEATURES))
    return out
