"""Regime-exit trade return engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_ml.config import BASE_COST, MAX_HOLD_DAYS, REGIME_CONF_THRESHOLD


def regime_exit_returns(
    signal: pd.Series,
    close: pd.Series,
    open_prices: pd.Series,
    regimes: pd.Series,
    probabilities: np.ndarray,
    start_pos: int,
    end_pos: int,
    slippage: pd.Series,
    strategy_name: str = "",
) -> pd.Series:
    """
    Compute sparse trade returns with regime-exit logic.

    Entry: signal at bar i → enter at open[i+1]
    Exit: first of (a) MAX_HOLD_DAYS or (b) regime change with
          P(entry regime) < 1 - τ
    Net return is stored on the entry bar; positions do not overlap.
    """
    returns = pd.Series(0.0, index=close.index)
    n = len(close)
    i = start_pos
    trade_count = 0

    while i <= end_pos:
        sig = float(signal.iloc[i])
        if sig == 0 or i + 1 >= n:
            i += 1
            continue

        entry_price = open_prices.iloc[i + 1]
        entry_regime = int(regimes.iloc[i])
        if pd.isna(entry_price) or entry_price == 0:
            i += 1
            continue

        exit_bar = min(i + MAX_HOLD_DAYS, end_pos, n - 1)
        for j in range(i + 1, min(i + MAX_HOLD_DAYS + 1, end_pos + 1)):
            current_regime = int(regimes.iloc[j])
            if current_regime != entry_regime:
                if 0 <= entry_regime < probabilities.shape[1]:
                    prob = probabilities[j, entry_regime]
                else:
                    prob = 0.0
                if prob < (1.0 - REGIME_CONF_THRESHOLD):
                    exit_bar = j
                    break

        exit_price = open_prices.iloc[exit_bar]
        if pd.isna(exit_price) or exit_price == 0:
            i = exit_bar + 1
            continue

        cost = 2 * BASE_COST + float(slippage.iloc[i]) + float(slippage.iloc[exit_bar])
        raw_return = (exit_price - entry_price) / entry_price
        returns.iloc[i] = sig * (raw_return - cost)
        trade_count += 1
        i = exit_bar + 1

    if strategy_name:
        if trade_count > 0:
            avg_hold = (end_pos - start_pos) / trade_count
            print(f"    [{strategy_name}] Trades: {trade_count}  Avg hold: {avg_hold:.1f} bars")
        else:
            print(f"    [{strategy_name}] Trades: 0")
    return returns


def sharpe_from_trades(return_series: pd.Series) -> float:
    """Sharpe on non-zero trade returns only (avoids sparse-zero deflation)."""
    trades = return_series[return_series != 0]
    if len(trades) < 3:
        return 0.0
    std = float(trades.std())
    if std == 0:
        return 0.0
    return float(trades.mean() / std)
