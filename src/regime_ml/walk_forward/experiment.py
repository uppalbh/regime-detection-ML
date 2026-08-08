"""Walk-forward experiment orchestration."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from regime_ml.backtesting.adaptive import RegimeBacktester, evaluate_returns
from regime_ml.config import (
    DEFAULT_WINDOWS,
    MAX_SHARPE_CAP,
    REGIME_FEATURES,
    TRADING_DAYS_PER_YEAR,
)
from regime_ml.costs.slippage import compute_slippage
from regime_ml.data.loader import Instrument
from regime_ml.features.regime_features import build_regime_features
from regime_ml.regimes.detector import RegimeDetector
from regime_ml.regimes.exits import regime_exit_returns
from regime_ml.strategies.signals import default_strategies


def _cap_sharpe(x: float) -> float:
    return float(max(min(x, MAX_SHARPE_CAP), -MAX_SHARPE_CAP))


def compute_test_returns(
    instrument: Instrument,
    test_data: pd.DataFrame,
    train_reg: RegimeDetector,
    test_regimes_arr: np.ndarray,
    test_probs_arr: np.ndarray,
) -> pd.DataFrame:
    """Regime-exit strategy returns over the test window using frozen GMM labels."""
    close = instrument.data["Close"]
    opens = instrument.data["Open"]
    full_idx = close.index
    slip = compute_slippage(close)

    start_pos = int(full_idx.searchsorted(test_data.index[0]))
    end_pos = int(full_idx.searchsorted(test_data.index[-1]))

    regime_arr = np.zeros(len(full_idx), dtype=int)
    for k, date in enumerate(train_reg.feature_data.index):
        pos = full_idx.searchsorted(date)
        if pos < len(full_idx):
            regime_arr[pos] = train_reg.regimes[k]
    for k, date in enumerate(test_data.index):
        pos = full_idx.searchsorted(date)
        if pos < len(full_idx):
            regime_arr[pos] = test_regimes_arr[k]
    regime_series = pd.Series(regime_arr, index=full_idx)

    k_regimes = test_probs_arr.shape[1]
    prob_arr = np.zeros((len(full_idx), k_regimes))
    for k, date in enumerate(test_data.index):
        pos = full_idx.searchsorted(date)
        if pos < len(full_idx):
            prob_arr[pos] = test_probs_arr[k]

    out = {}
    for name, strat in default_strategies().items():
        print(f"  {name}...")
        sig = strat.generate_signals(instrument.data)["Signal"].shift(1).fillna(0.0)
        sig = np.clip(sig, -1, 1)
        ret = regime_exit_returns(
            signal=sig,
            close=close,
            open_prices=opens,
            regimes=regime_series,
            probabilities=prob_arr,
            start_pos=start_pos,
            end_pos=end_pos,
            slippage=slip,
            strategy_name=name,
        )
        out[name] = ret.loc[test_data.index]

    return pd.DataFrame(out, index=test_data.index)


def buy_and_hold_metrics(close: pd.Series) -> dict:
    rets = close.pct_change().fillna(0.0)
    total = float((close.iloc[-1] - close.iloc[0]) / close.iloc[0])
    n = max(len(rets), 1)
    ann = float((1 + total) ** (TRADING_DAYS_PER_YEAR / n) - 1)
    vol = float(rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = float(ann / vol) if vol != 0 else 0.0
    return {"Sharpe": sharpe, "Ann_Return": ann, "Ann_Vol": vol, "n_days": n, "returns": rets}


def run_walk_forward(
    ticker: str,
    windows: Optional[list[tuple[str, str, str, str]]] = None,
    interval: str = "1d",
    *,
    force_download: bool = False,
) -> list[dict]:
    """
    Run walk-forward windows for one asset.

    For each window: fit GMM on train only, apply frozen GMM to test
    (no Hungarian alignment — labels stay consistent by construction).
    """
    windows = windows or DEFAULT_WINDOWS
    results = []

    for train_start, train_end, test_start, test_end in windows:
        print(f"\n{'=' * 55}")
        print(f"  Train: {train_start} → {train_end}")
        print(f"  Test:  {test_start} → {test_end}")
        print("=" * 55)

        train_years = (pd.Timestamp(train_end) - pd.Timestamp(train_start)).days / 365
        period_str = f"{max(int(train_years) + 2, 4)}y"

        stock = Instrument(ticker)
        stock.fetch(period_str, interval, force_download=force_download)
        features = build_regime_features(stock.data)
        stock.features = features

        train_df = features[(features.index >= train_start) & (features.index <= train_end)]
        test_df = features[(features.index >= test_start) & (features.index <= test_end)]

        if len(train_df) < 50 or len(test_df) < 20:
            print("  Skipping — insufficient data")
            continue

        reg = RegimeDetector(stock, period_str, interval)
        reg.feature_data = train_df.copy()

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_df[REGIME_FEATURES])
        reg.scaler = scaler
        reg.find_optimal_k(x_train)
        reg.fit_gmm(x_train)
        reg.regimes = reg.gmm.predict(x_train)
        reg.probabilities = reg.gmm.predict_proba(x_train)
        reg.interpret_regimes()
        reg.strategy_by_regime()

        # Frozen GMM on test — no label remapping needed
        x_test = scaler.transform(test_df[REGIME_FEATURES])
        test_regimes = reg.gmm.predict(x_test)
        test_probs = reg.gmm.predict_proba(x_test)

        test_returns = compute_test_returns(stock, test_df, reg, test_regimes, test_probs)

        bt = RegimeBacktester(reg, test_returns, test_regimes, test_probs)
        bt.run_adaptive()
        bt.run_equal_weight()
        bt.run_static()

        close_test = stock.data["Close"].reindex(test_df.index).dropna()
        bh = buy_and_hold_metrics(close_test)

        adaptive_sh = bt.adaptive_metrics["Sharpe"]
        ew_sh = bt.ew_metrics["Sharpe"]
        bh_sharpe = bh["Sharpe"]

        print(f"\n  Buy-and-Hold Sharpe : {bh_sharpe:.2f}")
        print(f"  Adaptive Sharpe     : {adaptive_sh:.2f}  (capped: {_cap_sharpe(adaptive_sh):.2f})")
        print(f"  Equal-Weight Sharpe : {ew_sh:.2f}  (capped: {_cap_sharpe(ew_sh):.2f})")
        print(
            f"  Beats BH?  {'YES' if adaptive_sh > bh_sharpe else 'NO'}   "
            f"Beats EW?  {'YES' if adaptive_sh > ew_sh else 'NO'}"
        )

        results.append(
            {
                "ticker": ticker,
                "test_period": f"{test_start} to {test_end}",
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "bh_sharpe": round(_cap_sharpe(bh_sharpe), 3),
                "adaptive_sharpe": round(_cap_sharpe(adaptive_sh), 3),
                "ew_sharpe": round(_cap_sharpe(ew_sh), 3),
                "beats_bh": bool(adaptive_sh > bh_sharpe),
                "beats_ew": bool(adaptive_sh > ew_sh),
                "n_days": int(bh["n_days"]),
                "optimal_k": int(reg.optimal_k),
                "adaptive_metrics": bt.adaptive_metrics,
                "ew_metrics": bt.ew_metrics,
                "static_metrics": bt.static_metrics,
                "portfolio_returns": bt.portfolio_returns,
                "bh_returns": bh["returns"],
                "regime_detector": reg,
                "backtester": bt,
                "test_regimes": test_regimes,
            }
        )

    return results
