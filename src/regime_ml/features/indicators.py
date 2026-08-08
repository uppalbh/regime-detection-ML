"""Base technical indicators used by strategies and regime features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def add_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)
    out["Simple_Return"] = close.pct_change()
    out["SMA_20"] = sma(close, 20)
    out["SMA_50"] = sma(close, 50)
    out["EMA_20"] = ema(close, 20)
    out["EMA_50"] = ema(close, 50)
    out["Rolling_STD_20"] = out["Simple_Return"].rolling(20, min_periods=20).std()
    out["RSI_14"] = rsi(close, 14)
    out["ATR_14"] = atr(out, 14)
    out["Rolling_Max_20"] = close.rolling(20, min_periods=20).max()
    out["Rolling_Min_20"] = close.rolling(20, min_periods=20).min()
    out["Rolling_Max_50"] = close.rolling(50, min_periods=50).max()
    out["Rolling_Min_50"] = close.rolling(50, min_periods=50).min()

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    out["MACD_Line"] = macd_line
    out["MACD_Signal"] = signal
    out["MACD_Histogram"] = macd_line - signal

    mid = sma(close, 20)
    std = close.rolling(20, min_periods=20).std()
    out["Bollinger_Middle"] = mid
    out["Bollinger_Upper"] = mid + 2 * std
    out["Bollinger_Lower"] = mid - 2 * std
    return out
