"""Strategy signal generators used by the regime allocator."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from regime_ml.features import indicators as ind


def hold_until_flip(signal: pd.Series) -> pd.Series:
    return signal.replace(0.0, np.nan).ffill().fillna(0.0)


class Strategy(ABC):
    name: str = "strategy"

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with Signal in {-1, 0, 1} using close-of-bar info."""


class MovingAverageCrossover(Strategy):
    name = "MovingAvg"

    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["Close"]
        fast = ind.sma(close, self.fast_period)
        slow = ind.sma(close, self.slow_period)
        signal = pd.Series(0.0, index=close.index)
        signal = signal.mask(fast > slow, 1.0).mask(fast < slow, -1.0)
        signal = signal.where(fast.notna() & slow.notna(), 0.0)
        return pd.DataFrame({"Close": close, "Signal": signal})


class MomentumRSI(Strategy):
    name = "Momentum"

    def __init__(self, rsi_period: int = 14, overbought: float = 70.0, oversold: float = 30.0):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["Close"]
        rsi = ind.rsi(close, self.rsi_period)
        event = pd.Series(0.0, index=close.index)
        event = event.mask(rsi < self.oversold, 1.0).mask(rsi > self.overbought, -1.0)
        signal = hold_until_flip(event.where(rsi.notna(), 0.0))
        return pd.DataFrame({"Close": close, "RSI": rsi, "Signal": signal})


class MeanReversion(Strategy):
    name = "MeanRev"

    def __init__(self, window: int = 20, entry_z: float = 1.0):
        self.window = window
        self.entry_z = entry_z

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["Close"]
        mid = close.rolling(self.window, min_periods=self.window).mean()
        std = close.rolling(self.window, min_periods=self.window).std()
        z = (close - mid) / std.replace(0.0, np.nan)
        event = pd.Series(0.0, index=close.index)
        event = event.mask(z < -self.entry_z, 1.0).mask(z > self.entry_z, -1.0)
        held = hold_until_flip(event.where(z.notna(), 0.0))
        signal = held.where(z.abs() >= self.entry_z * 0.5, 0.0).where(z.notna(), 0.0)
        return pd.DataFrame({"Close": close, "Z_Score": z, "Signal": signal})


class Breakout(Strategy):
    name = "Breakout"

    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["Close"]
        roll_high = close.shift(1).rolling(self.lookback, min_periods=self.lookback).max()
        roll_low = close.shift(1).rolling(self.lookback, min_periods=self.lookback).min()
        event = pd.Series(0.0, index=close.index)
        event = event.mask(close > roll_high, 1.0).mask(close < roll_low, -1.0)
        valid = roll_high.notna() & roll_low.notna()
        signal = hold_until_flip(event.where(valid, 0.0))
        return pd.DataFrame({"Close": close, "Signal": signal})


class TrendFollow(Strategy):
    name = "TrendFollow"

    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["Close"]
        ema_fast = ind.ema(close, self.fast)
        ema_slow = ind.ema(close, self.slow)
        signal = pd.Series(0.0, index=close.index)
        bull = (ema_fast > ema_slow) & (close > ema_fast)
        bear = (ema_fast < ema_slow) & (close < ema_fast)
        signal = signal.mask(bull, 1.0).mask(bear, -1.0)
        signal = signal.where(ema_fast.notna() & ema_slow.notna(), 0.0)
        return pd.DataFrame({"Close": close, "Signal": signal})


def default_strategies() -> dict[str, Strategy]:
    return {
        "MovingAvg": MovingAverageCrossover(5, 20),
        "Momentum": MomentumRSI(14, 70, 30),
        "MeanRev": MeanReversion(20, 1.0),
        "Breakout": Breakout(50),
        "TrendFollow": TrendFollow(20, 50),
    }
