#!/usr/bin/env python3
"""Run walk-forward regime-adaptive strategy experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_ml.config import (  # noqa: E402
    ASSET_UNIVERSES,
    DEFAULT_WINDOWS,
    FIGURES_DIR,
    ensure_output_dirs,
)
from regime_ml.reporting.summary import export_tables, print_summary, results_to_frame  # noqa: E402
from regime_ml.visualization.plots import (  # noqa: E402
    plot_bic_curve,
    plot_equity_comparison,
    plot_price_regimes,
    plot_walk_forward_sharpes,
)
from regime_ml.walk_forward.experiment import run_walk_forward  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward regime detection experiments.")
    parser.add_argument(
        "--assets",
        default="demo",
        help="Comma-separated tickers, or one of: demo, core, mega_tech, crypto, all",
    )
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap on walk-forward windows (useful for smoke tests).",
    )
    return parser.parse_args()


def resolve_tickers(spec: str) -> list[str]:
    key = spec.strip().lower()
    if key == "demo":
        return ["SPY", "AAPL"]
    if key == "core":
        return list(ASSET_UNIVERSES["core_equity"])
    if key == "all":
        return (
            ASSET_UNIVERSES["mega_tech"]
            + ASSET_UNIVERSES["core_equity"]
            + ASSET_UNIVERSES["crypto"]
        )
    if key in ASSET_UNIVERSES:
        return list(ASSET_UNIVERSES[key])
    return [t.strip() for t in spec.split(",") if t.strip()]


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    tickers = resolve_tickers(args.assets)
    windows = DEFAULT_WINDOWS
    if args.max_windows is not None:
        windows = DEFAULT_WINDOWS[: max(args.max_windows, 1)]

    print("\n" + "=" * 80)
    print("  RUNNING WALK-FORWARD EXPERIMENTS")
    print(f"  Assets: {tickers}")
    print(f"  Windows: {len(windows)}")
    print("=" * 80)

    asset_results: dict[str, list] = {}
    for ticker in tickers:
        print(f"\n{'#' * 55}\n  {ticker}\n{'#' * 55}")
        try:
            asset_results[ticker] = run_walk_forward(
                ticker,
                windows,
                interval=args.interval,
                force_download=args.force_download,
            )
        except Exception as exc:  # keep other assets running
            print(f"  ERROR for {ticker}: {exc}")
            asset_results[ticker] = []

    print_summary(asset_results)
    paths = export_tables(asset_results)
    print("\nTables written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    # Figures from first successful result
    comparison = results_to_frame(asset_results)
    if not comparison.empty:
        plot_walk_forward_sharpes(comparison, FIGURES_DIR / "walk_forward_sharpes.png")

    for ticker, windows_out in asset_results.items():
        if not windows_out:
            continue
        first = windows_out[0]
        rd = first["regime_detector"]
        if rd.bic_scores is not None:
            plot_bic_curve(
                rd.bic_scores,
                rd.optimal_k,
                FIGURES_DIR / f"{ticker}_bic_curve.png",
            )
        close = rd.instrument.data["Close"].reindex(rd.feature_data.index)
        plot_price_regimes(
            close,
            rd.regimes,
            title=f"{ticker} Train Regimes (first window)",
            path=FIGURES_DIR / f"{ticker}_price_regimes.png",
        )
        last = windows_out[-1]
        bt = last["backtester"]
        ew_rets = bt.test_returns.mean(axis=1)
        plot_equity_comparison(
            last["portfolio_returns"],
            ew_rets,
            last["bh_returns"],
            title=f"{ticker} Equity — {last['test_period']}",
            path=FIGURES_DIR / f"{ticker}_equity_comparison.png",
        )
        break  # one representative asset is enough for default figures

    print(f"\nFigures → {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
