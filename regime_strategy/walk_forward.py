def run_walk_forward_with_comparison(windows, stock_name, interval_v, return_backtester=False):
    results = []
    last_backtester = None

    for train_start, train_end, test_start, test_end in windows:
        print(f"\n{'='*60}")
        print(f"Train: {train_start} to {train_end}")
        print(f"Test:  {test_start} to {test_end}")
        print('='*60)

        # Fetch training data
        stock = Stock(stock_name)
        train_years = (pd.Timestamp(train_end) - pd.Timestamp(train_start)).days / 365
        period_arg  = f"{max(int(train_years) + 2, 4)}y"
        stock.fetch_data(period_arg, interval_v)
        stock.calculate_all_indicators()

        meta = MetaModelDataset(stock)
        df   = meta.create_features()

        # Filter to business days
        df = df[df.index.dayofweek < 5]

        train_data = df[(df.index >= train_start) & (df.index <= train_end)]
        test_data  = df[(df.index >= test_start)  & (df.index <= test_end)]

        if len(train_data) < 50 or len(test_data) < 20:
            print("WARNING: Insufficient data — skipping.")
            continue

        # Fit regime detector on train data
        train_reg = RegimeDetector(stock, period_arg, interval_v)
        train_reg.feature_data = train_data

        scaler_tr      = StandardScaler()
        X_train_scaled = scaler_tr.fit_transform(train_data[train_reg.regime_feat])
        train_reg.x_scaled = X_train_scaled
        train_reg.scaler   = scaler_tr

        train_reg.find_optimal_regimes()
        train_reg.fit_regime_models()
        train_reg.interpret_regimes(train_data)

        # Build strategy to regime mapping
        train_reg.strategy_by_regime()

        # Apply trained GMM to test data
        X_test_scaled    = scaler_tr.transform(test_data[train_reg.regime_feat])
        test_regimes_raw = train_reg.gmmModel.predict(X_test_scaled)
        test_probs_raw   = train_reg.gmmModel.predict_proba(X_test_scaled)

        # Align labels
        test_regimes, test_probs = align_regime_labels_hungarian(
            train_reg, test_regimes_raw, test_probs_raw, test_data
        )

        # Calculate test period returns
        test_returns = calculate_test_strategy_returns(
            stock, test_data,
            train_reg.strategy_regime_results,
            train_reg, test_regimes, test_probs
        )

        # Configure backtester
        class TestWrapper:
            def __init__(self):
                self.regime_feat          = train_reg.regime_feat
                self.optimal_n            = train_reg.optimal_n
                self.strategy_regime_results = train_reg.strategy_regime_results
                self.regimes              = test_regimes
                self.probabilities        = test_probs
                self.feature_data         = test_data
                self.strategy_returns_df  = test_returns
                self.instrument           = stock

            def strategy_by_regime(self):
                return self.strategy_regime_results, pd.DataFrame()

        wrapper = TestWrapper()

        regB = RegimeBacktester(wrapper)
        regB.strategy_returns = test_returns
        regB.build_regime_weight()
        regB.build_daily_strategy_weights()
        regB.run_adaptive_portfolio()
        regB.run_equal_weight_portfolio()
        regB.run_static_portfolio()

        # Calculate buy and hold benchmark
        close_test    = stock.indicator_data["Close"].loc[test_data.index]
        bh_rets       = close_test.pct_change().fillna(0)
        total_bh      = (close_test.iloc[-1] - close_test.iloc[0]) / close_test.iloc[0]
        n_days        = len(bh_rets)
        ann_ret_bh    = (1 + total_bh) ** (252 / n_days) - 1
        ann_vol_bh    = bh_rets.std() * np.sqrt(252)
        sharpe_bh     = ann_ret_bh / ann_vol_bh if ann_vol_bh != 0 else 0

        # Collect metrics
        adaptive_sh   = regB.adaptive_results["Sharpe"]
        ew_sh         = regB.equal_weight_results["Sharpe"]

        MAX_CAP = 3.0
        def cap(x): return max(min(x, MAX_CAP), -MAX_CAP)

        print(f"\n  RESULTS:")
        print(f"    Buy-and-Hold Sharpe: {sharpe_bh:.2f}  (capped: {cap(sharpe_bh):.2f})")
        print(f"    Adaptive Sharpe:     {adaptive_sh:.2f}  (capped: {cap(adaptive_sh):.2f})")
        print(f"    Equal-Weight Sharpe: {ew_sh:.2f}  (capped: {cap(ew_sh):.2f})")
        print(f"    Adaptive beats Equal-Weight: {'YES' if adaptive_sh > ew_sh else 'NO'}")
        print(f"    Adaptive beats Buy-and-Hold: {'YES' if adaptive_sh > sharpe_bh else 'NO'}")

        results.append({
            "test_period":              f"{test_start} to {test_end}",
            "buy_hold_sharpe":          cap(sharpe_bh),
            "adaptive_sharpe":          cap(adaptive_sh),
            "equal_weight_sharpe":      cap(ew_sh),
            "adaptive_wins_vs_bh":      adaptive_sh > sharpe_bh,
            "adaptive_wins_vs_eq":      adaptive_sh > ew_sh,
            "test_days":                n_days,
        })
        
        # Store the last backtester instance
        last_backtester = regB

    if return_backtester:
        return results, last_backtester
    return results

def print_summary(asset_results_map):
    print("\n" + "="*65)
    print("FINAL SUMMARY: Adaptive vs Equal-Weight vs Buy-and-Hold")
    print("="*65)

    total_bh_wins = total_eq_wins = total_windows = 0

    # Handle case with no results
    if not asset_results_map:
        print("\nNo results to summarize. Run walk-forward tests first.")
        return

    for asset, results in asset_results_map.items():
        print(f"\n{asset}:")
        print("-"*65)
        hdr = f"{'Test Period':<28} {'BH':>6} {'Adapt':>7} {'EW':>7}  BH?  EW?"
        print(hdr)
        print("-"*65)
        bh_w = eq_w = 0
        for r in results:
            bh_tag = "YES" if r["adaptive_wins_vs_bh"] else "NO"
            eq_tag = "YES" if r["adaptive_wins_vs_eq"] else "NO"
            print(f"  {r['test_period']:<26} "
                  f"{r['buy_hold_sharpe']:>6.2f} "
                  f"{r['adaptive_sharpe']:>7.2f} "
                  f"{r['equal_weight_sharpe']:>7.2f}  {bh_tag:<4} {eq_tag}")
            if r["adaptive_wins_vs_bh"]: bh_w += 1
            if r["adaptive_wins_vs_eq"]: eq_w += 1
        n = len(results)
        print(f"\n  Adaptive beats Buy-and-Hold:  {bh_w}/{n} ({100*bh_w//n}%)")
        print(f"  Adaptive beats Equal-Weight:  {eq_w}/{n} ({100*eq_w//n}%)")
        total_bh_wins += bh_w
        total_eq_wins += eq_w
        total_windows += n

    print("\n" + "="*65)
    print("COMBINED ACROSS ALL ASSETS")
    print("="*65)
    
    # Avoid division by zero
    if total_windows > 0:
        print(f"  Beats Buy-and-Hold: {total_bh_wins}/{total_windows} "
              f"({100*total_bh_wins//total_windows}%)")
        print(f"  Beats Equal-Weight: {total_eq_wins}/{total_windows} "
              f"({100*total_eq_wins//total_windows}%)")

        if total_bh_wins / total_windows >= 0.60:
            print("\nRegime-exit adaptive strategy beats buy-and-hold in >=60% of windows")
        else:
            print("\nRegime-exit strategy does not yet consistently beat buy-and-hold")
    else:
        print("No windows to summarize.")
