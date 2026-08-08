"""Summary tables and CSV exports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from regime_ml.config import TABLES_DIR, ensure_output_dirs


def results_to_frame(asset_results: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for ticker, windows in asset_results.items():
        for r in windows:
            rows.append(
                {
                    "ticker": ticker,
                    "test_period": r["test_period"],
                    "bh_sharpe": r["bh_sharpe"],
                    "adaptive_sharpe": r["adaptive_sharpe"],
                    "ew_sharpe": r["ew_sharpe"],
                    "beats_bh": r["beats_bh"],
                    "beats_ew": r["beats_ew"],
                    "n_days": r["n_days"],
                    "optimal_k": r.get("optimal_k"),
                }
            )
    return pd.DataFrame(rows)


def print_summary(asset_results: dict[str, list[dict]]) -> None:
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY: Adaptive vs Equal-Weight vs Buy-and-Hold")
    print("=" * 65)

    total_bh = total_ew = total = 0
    for asset, results in asset_results.items():
        print(f"\n  {asset}")
        print(f"  {'Period':<28} {'BH':>6} {'Adapt':>7} {'EW':>7}  BH  EW")
        print(f"  {'-' * 60}")
        bh_w = ew_w = 0
        for r in results:
            b = "Y" if r["beats_bh"] else "N"
            e = "Y" if r["beats_ew"] else "N"
            print(
                f"  {r['test_period']:<28} "
                f"{r['bh_sharpe']:>6.2f} "
                f"{r['adaptive_sharpe']:>7.2f} "
                f"{r['ew_sharpe']:>7.2f}  {b}  {e}"
            )
            if r["beats_bh"]:
                bh_w += 1
            if r["beats_ew"]:
                ew_w += 1
        n = len(results)
        print(f"\n  Beats BH: {bh_w}/{n}   Beats EW: {ew_w}/{n}")
        total_bh += bh_w
        total_ew += ew_w
        total += n

    print("\n" + "=" * 65)
    pct_bh = 100 * total_bh // max(total, 1)
    pct_ew = 100 * total_ew // max(total, 1)
    print(f"  COMBINED — Beats BH: {total_bh}/{total} ({pct_bh}%)  Beats EW: {total_ew}/{total} ({pct_ew}%)")
    if total > 0 and total_ew / total >= 0.60:
        print("  Adaptive beats equal-weight in >=60% of windows")
    else:
        print("  Adaptive does not consistently beat equal-weight on this run")


def export_tables(
    asset_results: dict[str, list[dict]],
    *,
    tables_dir: Optional[Path] = None,
) -> dict[str, Path]:
    ensure_output_dirs()
    out_dir = tables_dir or TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = results_to_frame(asset_results)
    paths = {}
    paths["walk_forward_comparison"] = out_dir / "walk_forward_comparison.csv"
    comparison.to_csv(paths["walk_forward_comparison"], index=False)

    # Aggregate win rates by ticker
    if not comparison.empty:
        agg = (
            comparison.groupby("ticker")
            .agg(
                windows=("test_period", "count"),
                beats_bh=("beats_bh", "sum"),
                beats_ew=("beats_ew", "sum"),
                mean_adaptive_sharpe=("adaptive_sharpe", "mean"),
                mean_bh_sharpe=("bh_sharpe", "mean"),
                mean_ew_sharpe=("ew_sharpe", "mean"),
            )
            .reset_index()
        )
        paths["asset_summary"] = out_dir / "asset_summary.csv"
        agg.to_csv(paths["asset_summary"], index=False)

    # Regime metrics from first successful window per ticker
    regime_rows = []
    for ticker, windows in asset_results.items():
        if not windows:
            continue
        rd = windows[0].get("regime_detector")
        if rd is None or rd.regime_metrics is None:
            continue
        m = rd.regime_metrics.copy()
        m.insert(0, "ticker", ticker)
        regime_rows.append(m)
    if regime_rows:
        regime_df = pd.concat(regime_rows, ignore_index=True)
        paths["regime_strategy_metrics"] = out_dir / "regime_strategy_metrics.csv"
        regime_df.to_csv(paths["regime_strategy_metrics"], index=False)

    # Static vs adaptive metrics for last window of each ticker
    static_rows = []
    for ticker, windows in asset_results.items():
        if not windows:
            continue
        last = windows[-1]
        static = last.get("static_metrics")
        if static is not None:
            tmp = static.copy()
            tmp["ticker"] = ticker
            tmp["test_period"] = last["test_period"]
            tmp["strategy"] = tmp.index
            static_rows.append(tmp.reset_index(drop=True))
        if last.get("adaptive_metrics"):
            static_rows.append(
                pd.DataFrame(
                    [
                        {
                            **last["adaptive_metrics"],
                            "ticker": ticker,
                            "test_period": last["test_period"],
                            "strategy": "Adaptive",
                        }
                    ]
                )
            )
        if last.get("ew_metrics"):
            static_rows.append(
                pd.DataFrame(
                    [
                        {
                            **last["ew_metrics"],
                            "ticker": ticker,
                            "test_period": last["test_period"],
                            "strategy": "EqualWeight",
                        }
                    ]
                )
            )
    if static_rows:
        paths["metrics"] = out_dir / "metrics.csv"
        pd.concat(static_rows, ignore_index=True).to_csv(paths["metrics"], index=False)

    return paths
