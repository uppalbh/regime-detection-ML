
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


def calculate_test_strategy_returns(stock, test_data, train_strategy_results,
                                    train_reg, test_regimes, test_probabilities):
    """
    Compute per-strategy returns for the TEST window using regime-exit logic.

    Parameters
    ----------
    stock                  : Stock object (full history fetched)
    test_data              : DataFrame — test-window feature rows
    train_strategy_results : pd.DataFrame — train-only regime×strategy metrics
    train_reg              : RegimeDetector fitted on train data
    test_regimes           : np.ndarray of regime labels for test bars
    test_probabilities     : np.ndarray (n_test, k) of regime probabilities
    """
    strategies = {
        "MovingAvg": MovingAverageCrossover(stock, fast_period=5, slow_period=20),
        "Momentum":  Momentum(stock),
        "MeanRev":   MeanReversion(stock),
        "Breakout":  Breakout(stock),
        "TrendFollow": TrendFollow(stock, fast=20, slow=50),
    }

    close       = stock.indicator_data["Close"]
    open_prices = stock.indicator_data["Open"]
    full_index  = close.index

    slip_series = compute_slippage_series(close)

    # Locate test window start / end in full index
    test_start_date = test_data.index[0]
    test_end_date   = test_data.index[-1]

    try:
        start_pos = full_index.get_loc(test_start_date)
    except KeyError:
        start_pos = full_index.searchsorted(test_start_date)

    try:
        end_pos = full_index.get_loc(test_end_date)
    except KeyError:
        end_pos = full_index.searchsorted(test_end_date)

    # Build regime_series for full index
    # For positions before test window: use train regime (from train_reg)
    # For test positions: use test_regimes
    regime_array = np.zeros(len(full_index), dtype=int)

    # Fill train portion from train_reg regimes (aligned to feature_data index)
    if train_reg.regimes is not None and train_reg.feature_data is not None:
        for k, date in enumerate(train_reg.feature_data.index):
            try:
                pos = full_index.get_loc(date)
                regime_array[pos] = train_reg.regimes[k]
            except KeyError:
                pass

    # Fill test portion
    for k, date in enumerate(test_data.index):
        try:
            pos = full_index.get_loc(date)
            regime_array[pos] = test_regimes[k]
        except KeyError:
            pass

    regime_series_full = pd.Series(regime_array, index=full_index)

    # Build probability array for full index (needed for threshold check)
    n_regimes = test_probabilities.shape[1]
    prob_array = np.zeros((len(full_index), n_regimes))
    for k, date in enumerate(test_data.index):
        try:
            pos = full_index.get_loc(date)
            prob_array[pos] = test_probabilities[k]
        except KeyError:
            pass

    returns_dict = {}

    for name, strategy in strategies.items():
        print(f"  Calculating {name} returns...")

        sig = strategy.generate_signals()["Signal"].shift(1).fillna(0)
        sig = np.clip(sig, -1, 1)

        ret_series = regime_exit_returns(
            signal_series   = sig,
            close_series    = close,
            open_series     = open_prices,
            regime_series   = regime_series_full,
            probabilities   = prob_array,
            full_index      = full_index,
            start_pos       = start_pos,
            end_pos         = end_pos,
            slippage_series = slip_series,
            strategy_name   = name,
        )
        returns_dict[name] = ret_series.loc[test_data.index]

    result_df = pd.DataFrame(returns_dict, index=test_data.index)
    print(f"  Result shape: {result_df.shape}")
    print(f"  Non-zero returns: {(result_df != 0).sum().sum()}")
    return result_df



