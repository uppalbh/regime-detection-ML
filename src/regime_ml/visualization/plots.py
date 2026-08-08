"""Plot helpers for regime detection research."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regime_ml.config import FIGURES_DIR, ensure_output_dirs


def _save(fig: plt.Figure, path: Path) -> Path:
    ensure_output_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bic_curve(bic_scores: list[float], optimal_k: int, path: Optional[Path] = None) -> Path:
    ks = list(range(2, 2 + len(bic_scores)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, bic_scores, marker="o", color="#1565c0")
    ax.axvline(optimal_k, color="#c62828", linestyle="--", label=f"Selected K={optimal_k}")
    ax.set_title("GMM BIC Curve")
    ax.set_xlabel("Number of regimes (K)")
    ax.set_ylabel("BIC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path or FIGURES_DIR / "bic_curve.png")


def plot_price_regimes(
    close: pd.Series,
    regimes: np.ndarray,
    *,
    title: str,
    path: Optional[Path] = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    sc = ax.scatter(close.index, close.values, c=regimes, cmap="tab10", s=8, zorder=3)
    ax.plot(close.index, close.values, color="gray", alpha=0.35, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    fig.colorbar(sc, ax=ax, label="Regime")
    ax.grid(True, alpha=0.3)
    return _save(fig, path or FIGURES_DIR / "price_regimes.png")


def plot_equity_comparison(
    adaptive: pd.Series,
    equal_weight: pd.Series,
    buy_hold: pd.Series,
    *,
    title: str = "Equity Comparison",
    path: Optional[Path] = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, series in (
        ("Adaptive", adaptive),
        ("EqualWeight", equal_weight),
        ("BuyHold", buy_hold),
    ):
        equity = (1 + series.fillna(0.0)).cumprod()
        ax.plot(equity.index, equity.values, label=name, linewidth=1.3)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path or FIGURES_DIR / "equity_comparison.png")


def plot_walk_forward_sharpes(comparison: pd.DataFrame, path: Optional[Path] = None) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(comparison))
    width = 0.25
    ax.bar(x - width, comparison["bh_sharpe"], width, label="Buy&Hold")
    ax.bar(x, comparison["adaptive_sharpe"], width, label="Adaptive")
    ax.bar(x + width, comparison["ew_sharpe"], width, label="EqualWeight")
    labels = [f"{r.ticker}\n{r.test_period.split(' to ')[0][2:]}" for r in comparison.itertuples()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Capped Sharpe")
    ax.set_title("Walk-Forward Sharpe by Window")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, path or FIGURES_DIR / "walk_forward_sharpes.png")
