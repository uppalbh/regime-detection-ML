"""Gaussian Mixture Model regime detector."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from regime_ml.config import REGIME_FEATURES, TRAIN_RATIO
from regime_ml.costs.slippage import compute_slippage
from regime_ml.data.loader import Instrument
from regime_ml.features.regime_features import build_regime_features
from regime_ml.regimes.exits import regime_exit_returns, sharpe_from_trades
from regime_ml.strategies.signals import default_strategies


class RegimeDetector:
    """
    Detect market regimes with a GMM fit on training data only.

    Pipeline: features → BIC K selection → fit GMM → regime profiles →
    train-period strategy-to-regime Sharpe weights.
    """

    def __init__(self, instrument: Instrument, period: str, interval: str):
        self.instrument = instrument
        self.period = period
        self.interval = interval

        self.feature_data: Optional[pd.DataFrame] = None
        self.feature_data_with_regime: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None
        self.gmm: Optional[GaussianMixture] = None
        self.optimal_k: Optional[int] = None
        self.regimes: Optional[np.ndarray] = None
        self.probabilities: Optional[np.ndarray] = None
        self.regime_prof: Optional[pd.DataFrame] = None
        self.bic_scores: Optional[list[float]] = None
        self.switch_rate: Optional[float] = None
        self.avg_confidence: Optional[float] = None

        self.strategy_returns: Optional[pd.DataFrame] = None
        self.strategy_regime_map: Optional[dict] = None
        self.regime_metrics: Optional[pd.DataFrame] = None

    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
        clean = df[REGIME_FEATURES].dropna()
        self.feature_data = df.loc[clean.index].copy()
        scaler = StandardScaler()
        x = scaler.fit_transform(clean)
        self.scaler = scaler
        return x, scaler

    def find_optimal_k(self, x: np.ndarray, max_k: int = 8) -> int:
        bic_scores = []
        for k in range(2, max_k + 1):
            gm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
            gm.fit(x)
            bic_scores.append(gm.bic(x))

        total_range = bic_scores[0] - min(bic_scores)
        optimal_k = len(bic_scores) + 1
        for i in range(1, len(bic_scores)):
            improvement = bic_scores[i - 1] - bic_scores[i]
            if total_range > 0 and improvement / total_range < 0.05:
                optimal_k = i + 1  # k starts at 2
                break

        self.optimal_k = max(2, min(optimal_k, 6))
        self.bic_scores = bic_scores
        print(f"BIC selected K = {self.optimal_k}")
        return self.optimal_k

    def fit_gmm(self, x: np.ndarray) -> GaussianMixture:
        gmm = GaussianMixture(
            n_components=self.optimal_k,
            covariance_type="full",
            random_state=42,
            n_init=10,
        )
        gmm.fit(x)
        self.gmm = gmm
        self.regimes = gmm.predict(x)
        self.probabilities = gmm.predict_proba(x)
        return gmm

    def interpret_regimes(self) -> pd.DataFrame:
        df = self.feature_data.copy()
        df["regime"] = self.regimes
        self.feature_data_with_regime = df
        self.regime_prof = df.groupby("regime")[REGIME_FEATURES].mean()
        print("\nRegime Feature Means:")
        print(self.regime_prof.round(4))
        return self.regime_prof

    def analyze_persistence(self) -> float:
        changes = int(np.sum(self.regimes[1:] != self.regimes[:-1]))
        switch_rate = changes / len(self.regimes)
        self.switch_rate = switch_rate
        quality = (
            "Excellent" if switch_rate < 0.10 else "Good" if switch_rate < 0.20 else "Poor"
        )
        print(f"Regime Persistence: {quality} (switch_rate = {switch_rate:.3f})")
        return switch_rate

    def analyze_confidence(self) -> float:
        avg = float(self.probabilities.max(axis=1).mean())
        self.avg_confidence = avg
        quality = (
            "Excellent"
            if avg > 0.90
            else "Good"
            if avg > 0.75
            else "Moderate"
            if avg > 0.60
            else "Heavy overlap"
        )
        print(f"GMM Confidence: {avg:.3f} ({quality})")
        return avg

    def run(self, train_only: bool = True) -> "RegimeDetector":
        self.instrument.fetch(self.period, self.interval)
        features = build_regime_features(self.instrument.data)
        self.instrument.features = features

        if train_only:
            split = int(len(features) * TRAIN_RATIO)
            train_df = features.iloc[:split]
            x_train, _ = self.prepare_features(train_df)
            self.feature_data = train_df.copy()
        else:
            x_train, _ = self.prepare_features(features)

        self.find_optimal_k(x_train)
        self.fit_gmm(x_train)
        self.interpret_regimes()
        return self

    def strategy_by_regime(self) -> pd.DataFrame:
        """Map strategies to regimes using train-period regime-exit Sharpes."""
        idx = self.feature_data.index
        close = self.instrument.data["Close"].reindex(idx)
        opens = self.instrument.data["Open"].reindex(idx)
        slip = compute_slippage(self.instrument.data["Close"]).reindex(idx).ffill()
        regimes = pd.Series(self.regimes, index=idx)

        # Signals use full OHLCV history, then align to feature index
        strategies = default_strategies()
        returns_dict = {}
        n = len(self.feature_data)

        for name, strat in strategies.items():
            print(f"  Computing {name}...")
            raw = strat.generate_signals(self.instrument.data)["Signal"].shift(1).fillna(0.0)
            sig = np.clip(raw, -1, 1).reindex(idx).fillna(0.0)
            returns_dict[name] = regime_exit_returns(
                signal=sig,
                close=close,
                open_prices=opens,
                regimes=regimes,
                probabilities=self.probabilities,
                start_pos=0,
                end_pos=n - 1,
                slippage=slip,
                strategy_name=name,
            )

        self.strategy_returns = pd.DataFrame(returns_dict, index=idx)

        df_reg = self.feature_data.copy()
        df_reg["regime"] = self.regimes
        results = []
        for regime_id in sorted(df_reg["regime"].unique()):
            subset = df_reg[df_reg["regime"] == regime_id]
            for s in strategies:
                rets = self.strategy_returns.loc[subset.index, s]
                trades = rets[rets != 0]
                results.append(
                    {
                        "Regime": regime_id,
                        "Strategy": s,
                        "Sharpe": sharpe_from_trades(rets),
                        "Win_Rate": float((trades > 0).mean()) if len(trades) else 0.0,
                        "Trades": int(len(trades)),
                    }
                )

        self.regime_metrics = pd.DataFrame(results)
        self.strategy_regime_map = {}
        for regime_id in sorted(df_reg["regime"].unique()):
            subset = self.regime_metrics[self.regime_metrics["Regime"] == regime_id]
            pos = subset[subset["Sharpe"] > 0]
            total_sh = pos["Sharpe"].sum()
            weights = {}
            for _, row in subset.iterrows():
                sh = row["Sharpe"]
                weights[row["Strategy"]] = (
                    float(sh / total_sh) if sh > 0 and total_sh > 0 else 0.0
                )
            best = subset.sort_values("Sharpe", ascending=False).iloc[0]
            self.strategy_regime_map[int(regime_id)] = {
                "weights": weights,
                "best_strategy": best["Strategy"],
                "best_sharpe": float(best["Sharpe"]),
            }
        return self.regime_metrics
