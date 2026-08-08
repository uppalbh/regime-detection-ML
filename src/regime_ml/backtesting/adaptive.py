"""Adaptive regime-aware portfolio backtester."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from regime_ml.config import TRADING_DAYS_PER_YEAR
from regime_ml.regimes.detector import RegimeDetector


def evaluate_returns(returns: pd.Series) -> dict:
    """Annualized metrics on a (possibly sparse) return series."""
    n = max(len(returns), 1)
    total_ret = float((1 + returns).prod() - 1)
    ann_ret = float((1 + total_ret) ** (TRADING_DAYS_PER_YEAR / n) - 1)
    ann_vol = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = float(ann_ret / ann_vol) if ann_vol != 0 else 0.0
    equity = (1 + returns).cumprod()
    max_dd = float(((equity - equity.cummax()) / equity.cummax()).min())
    win_rate = float((returns > 0).mean())
    return {
        "Sharpe": round(sharpe, 3),
        "Ann_Return": round(ann_ret, 4),
        "Ann_Vol": round(ann_vol, 4),
        "Max_Drawdown": round(max_dd, 4),
        "Win_Rate": round(win_rate, 4),
    }


class RegimeBacktester:
    """Probability-weighted blend of strategy returns using regime map."""

    def __init__(
        self,
        regime_detector: RegimeDetector,
        test_returns: pd.DataFrame,
        test_regimes: np.ndarray,
        test_probs: np.ndarray,
    ):
        self.rd = regime_detector
        self.test_returns = test_returns
        self.test_regimes = test_regimes
        self.test_probs = test_probs

        self.daily_weights: Optional[pd.DataFrame] = None
        self.portfolio_returns: Optional[pd.Series] = None
        self.adaptive_metrics: Optional[dict] = None
        self.ew_metrics: Optional[dict] = None
        self.static_metrics: Optional[pd.DataFrame] = None

    def build_daily_weights(self) -> pd.DataFrame:
        """w_s(t) = Σ_k P(k|x_t) * weight(s, k)"""
        strat_names = list(self.test_returns.columns)
        rows = []
        for day_probs in self.test_probs:
            day_w = {}
            for s in strat_names:
                w = 0.0
                for k, prob in enumerate(day_probs):
                    if k in self.rd.strategy_regime_map:
                        w += float(prob) * self.rd.strategy_regime_map[k]["weights"].get(s, 0.0)
                day_w[s] = w
            rows.append(day_w)
        self.daily_weights = pd.DataFrame(rows, index=self.test_returns.index)
        return self.daily_weights

    def run_adaptive(self) -> pd.Series:
        if self.daily_weights is None:
            self.build_daily_weights()
        self.portfolio_returns = (self.daily_weights * self.test_returns).sum(axis=1)
        self.adaptive_metrics = evaluate_returns(self.portfolio_returns)
        return self.portfolio_returns

    def run_equal_weight(self) -> dict:
        ew = self.test_returns.mean(axis=1)
        self.ew_metrics = evaluate_returns(ew)
        return self.ew_metrics

    def run_static(self) -> pd.DataFrame:
        results = {col: evaluate_returns(self.test_returns[col]) for col in self.test_returns}
        self.static_metrics = pd.DataFrame(results).T
        return self.static_metrics

    def significance_test(self) -> tuple[float, float]:
        if self.portfolio_returns is None:
            raise RuntimeError("Run adaptive first.")
        if self.static_metrics is None:
            self.run_static()
        best_static = self.static_metrics["Sharpe"].idxmax()
        static_rets = self.test_returns[best_static].loc[self.portfolio_returns.index]
        n = min(len(self.portfolio_returns), len(static_rets))
        t_stat, p_val = ttest_rel(self.portfolio_returns.iloc[:n], static_rets.iloc[:n])
        print(f"Paired t-test vs {best_static}: t={t_stat:.3f}, p={p_val:.4f}")
        return float(t_stat), float(p_val)
