# Regime Detection ML

GMM-based market regime detection with adaptive strategy allocation,
walk-forward validation, and regime-exit execution.

Undergraduate / entry-level quant internship level research codebase.

## Purpose

This project asks whether market regimes estimated from a small causal feature
set can improve allocation across simple technical strategies relative to
equal-weight and buy-and-hold baselines.

## Research Motivation

Trend, momentum, mean-reversion, and breakout strategies perform differently
across bull, bear, and high-volatility states. A Gaussian Mixture Model (GMM)
on normalized regime features provides a probabilistic state estimate. Strategy
weights are learned on training windows only and applied out-of-sample with a
frozen GMM (no Hungarian label alignment).

## Architecture

```
DATA → REGIME FEATURES → GMM REGIMES → STRATEGY SIGNALS
     → REGIME-EXIT RETURNS → ADAPTIVE WEIGHTS → WALK-FORWARD EVALUATION
```

## Repository Structure

```
src/regime_ml/
  config.py
  data/             # OHLCV download/cache/clean
  features/         # indicators + 4 regime features
  strategies/       # MA, RSI, mean-reversion, breakout, trend
  costs/            # variance-scaled slippage
  regimes/          # GMM detector + regime-exit engine
  backtesting/      # adaptive / EW / static evaluation
  walk_forward/     # multi-window orchestrator
  reporting/        # tables + console summary
  visualization/    # BIC, regimes, equity plots
scripts/run_walk_forward.py
tests/
data/{raw,processed}/
outputs/{figures,tables}/
```

## Data

- Source: Yahoo Finance via `yfinance`
- Cache: `data/raw/{SYMBOL}_{PERIOD}_{INTERVAL}.csv`
- Cleaning: sort, dedupe timestamps, forward-fill, drop invalid OHLC
- Business-day filter applied to regime feature matrix

## Regime Features

| Feature | Role |
| --- | --- |
| `Trend_Normalized` | EMA distance scaled by vol regime |
| `Momentum_Normalized` | RSI scaled by vol regime |
| `Volatility_Stability` | Stability of the vol-regime ratio |
| `Vol_Change` | ATR expansion / contraction |

All features are causal (rolling / lag-based).

## Strategies

Signals are generated from close-of-bar information and shifted by one bar
before entry. Strategies: MovingAvg, Momentum (RSI), MeanRev, Breakout,
TrendFollow.

## Regime-Exit Execution

- Entry at next open after a non-zero signal
- Exit on first of: max hold (`60` bars) or regime change with
  `P(entry regime) < 1 - τ` (`τ = 0.75`)
- Round-trip cost: `2 * BASE_COST + slippage_entry + slippage_exit`
- Slippage: `α * 20-day return variance` (`α = 0.9`)
- No overlapping positions

## Adaptive Allocation

Train-only Sharpe (on non-zero trade returns) produces positive-Sharpe weights
per regime. At test time:

```
w_s(t) = Σ_k P(regime=k | x_t) * weight(s, k)
```

## Walk-Forward Design

Default windows (train → test):

1. 2017–2021 → 2022
2. 2018–2022 → 2023
3. 2019–2023 → 2024
4. 2020–2024 → 2025
5. 2021–2025 → 2026H1

Frozen train GMM is applied to each test window. Labels remain consistent by
construction (Hungarian alignment removed intentionally).

## Installation

```bash
cd regime-detection-ML
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execution

Demo run (SPY + AAPL, all windows):

```bash
PYTHONPATH=src python scripts/run_walk_forward.py --assets demo
```

Smoke test (one window):

```bash
PYTHONPATH=src python scripts/run_walk_forward.py --assets SPY --max-windows 1
```

Full thesis universe:

```bash
PYTHONPATH=src python scripts/run_walk_forward.py --assets all
```

Tests:

```bash
PYTHONPATH=src pytest -q
```

## Outputs

Tables under `outputs/tables/`:

- `walk_forward_comparison.csv`
- `asset_summary.csv`
- `regime_strategy_metrics.csv`
- `metrics.csv`

Figures under `outputs/figures/`:

- BIC curve, price-colored regimes, equity comparison, walk-forward Sharpes

## Assumptions

- Daily bars; 252-day annualization
- Open fills after close signals
- Sharpe caps at ±3 for summary tables (raw metrics kept in objects)
- Single-asset experiments (no cross-sectional portfolio)

## Limitations

- Sparse regime-exit returns make calendar Sharpe sensitive to trade count
- GMM assumes elliptical mixture components
- Yahoo history can be revised
- No borrow fees / overnight financing model
- Demo defaults intentionally smaller than the full 9-asset thesis sweep

## Future Improvements

- Multi-asset portfolio construction
- Purged walk-forward / combinatorial CV
- Ablation on τ, max hold, and feature set
- Formal statistical battery beyond paired t-tests

## License

See `LICENSE`.
