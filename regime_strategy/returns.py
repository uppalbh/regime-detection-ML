
def regime_exit_returns(
    signal_series,        # pd.Series of {-1, 0, 1}
    close_series,         # pd.Series of close prices (full instrument history)
    open_series,          # pd.Series of open prices  (full instrument history)
    regime_series,        # pd.Series of integer regime labels (full instrument history)
    probabilities,        # np.ndarray (n_bars, n_regimes) — same length as full index
    full_index,           # DatetimeIndex of full instrument history
    start_pos,            # int — first bar allowed to enter (test window start)
    end_pos,              # int — last bar in test window (inclusive)
    slippage_series,      # pd.Series aligned to full_index
    strategy_name="",
):
    """
    Compute per-bar returns using REGIME-EXIT logic.

    Entry:  signal fires at bar i  → buy at open[i+1]
    Exit:   whichever comes first —
              (a) regime changes (dominant regime shifts away from entry regime)
              (b) MAX_HOLD_DAYS bars have elapsed
              (c) signal reverses sign
    Return is recorded at bar i (entry bar), not exit bar.
    This preserves the sparse-series structure the backtester expects.
    """
    strategy_return = pd.Series(0.0, index=full_index)

    n = len(full_index)
    i = start_pos
    trade_count = 0

    while i <= end_pos:
        sig = signal_series.iloc[i]

        if sig == 0:
            i += 1
            continue

        # ── ENTRY ────────────────────────────────────────────
        if i + 1 >= n:
            break

        entry_price = open_series.iloc[i + 1]
        if pd.isna(entry_price) or entry_price == 0:
            i += 1
            continue

        entry_regime = regime_series.iloc[i]

        # ── SEARCH FOR EXIT ──────────────────────────────────
        exit_bar = None

        for j in range(i + 1, min(i + MAX_HOLD_DAYS + 1, end_pos + 1)):
            # (a) Regime change ONLY — signal flips within a regime are noise
            dominant_regime = regime_series.iloc[j]
            if dominant_regime != entry_regime:
                prob_entry_regime = probabilities[j, entry_regime] if probabilities is not None else 0
                if prob_entry_regime < (1 - REGIME_CHANGE_THRESHOLD):
                    exit_bar = j
                    break

        # (c) Max hold cap
        if exit_bar is None:
            exit_bar = min(i + MAX_HOLD_DAYS, end_pos, n - 1)

        if exit_bar >= n:
            break

        exit_price = open_series.iloc[exit_bar]
        if pd.isna(exit_price) or exit_price == 0:
            i = exit_bar + 1
            continue

        # ── COST ─────────────────────────────────────────────
        slippage_entry = slippage_series.iloc[i]
        slippage_exit  = slippage_series.iloc[exit_bar]
        total_cost = BASE_COST + slippage_entry + BASE_COST + slippage_exit  # round-trip

        raw_ret = (exit_price - entry_price) / entry_price
        net_ret = sig * (raw_ret - total_cost)

        strategy_return.iloc[i] = net_ret
        trade_count += 1

        # Jump to bar after exit (no overlapping positions)
        i = exit_bar + 1

    print(f"    [{strategy_name}] Trades: {trade_count}  "
          f"(avg hold: {f'{(end_pos - start_pos) / max(trade_count, 1):.1f} bars' if trade_count else 'N/A'})")
    return strategy_return

