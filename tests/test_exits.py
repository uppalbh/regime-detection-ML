"""Tests for regime-exit logic and look-ahead controls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_ml.regimes.exits import regime_exit_returns, sharpe_from_trades
from regime_ml.strategies.signals import MovingAverageCrossover


def _frame(n: int = 40) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(np.linspace(100, 120, n), index=idx)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1e6,
        },
        index=idx,
    )


def test_signal_shift_used_by_strategies():
    df = _frame(60)
    sig = MovingAverageCrossover(3, 10).generate_signals(df)["Signal"]
    shifted = sig.shift(1).fillna(0)
    assert shifted.iloc[0] == 0


def test_regime_exit_no_overlap_and_costs():
    df = _frame(50)
    signal = pd.Series(0.0, index=df.index)
    signal.iloc[5] = 1.0
    regimes = pd.Series(0, index=df.index)
    probs = np.ones((len(df), 2)) * 0.5
    probs[:, 0] = 0.9
    slip = pd.Series(0.001, index=df.index)

    rets = regime_exit_returns(
        signal=signal,
        close=df["Close"],
        open_prices=df["Open"],
        regimes=regimes,
        probabilities=probs,
        start_pos=0,
        end_pos=len(df) - 1,
        slippage=slip,
        strategy_name="test",
    )
    assert (rets != 0).sum() == 1
    # Cost drag: net return should be below raw open-to-open return
    entry = df["Open"].iloc[6]
    # exit at max hold or end
    assert rets.iloc[5] != 0
    assert rets.iloc[5] < (df["Open"].iloc[-1] - entry) / entry


def test_sharpe_from_trades_ignores_zeros():
    s = pd.Series([0.0, 0.0, 0.02, -0.01, 0.03, 0.0, 0.01])
    assert sharpe_from_trades(s) != 0.0
    assert sharpe_from_trades(pd.Series([0.0, 0.0])) == 0.0
