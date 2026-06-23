
class RegimeDetector:

    def __init__(self, instrument, period, interval,
                 regime_feat=("Trend", "Volatility_Regime", "Momentum",
                              "Momentum_Signal", "Vol_Change")):
        self.instrument    = instrument
        self.period        = period
        self.interval      = interval
        self.regime_feat   = list(regime_feat)

        self.feature_data  = None
        self.x_scaled      = None
        self.scaler        = None
        self.optimal_n     = None
        self.bic_scores    = None
        self.gmmModel      = None
        self.regimes       = None          # np.ndarray aligned to feature_data
        self.probabilities = None          # np.ndarray (n, k)
        self.regime_prof   = None
        self.regime_changes = None
        self.switch_rate   = None
        self.avg_confidence = None

        self.strategy_regime_results = None   # train-only metrics DataFrame
        self.strategy_regime_max     = None
        self.strategy_returns_df     = None   # full returns used by backtester
        self.train_df  = None
        self.test_df   = None

    # ── FEATURE PREP ─────────────────────────────────────────

    def prepare_regime_features(self, df):
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(df[self.regime_feat])
        self.x_scaled = X_scaled
        self.scaler   = scaler
        return X_scaled, scaler

    # ── OPTIMAL N ────────────────────────────────────────────

    def find_optimal_regimes(self, max_regimes=8):
        bic_scores = []
        for n in range(1, max_regimes + 1):
            gm = GaussianMixture(n_components=n, random_state=42)
            gm.fit(self.x_scaled)
            bic_scores.append(gm.bic(self.x_scaled))

        # Elbow detection: first point where improvement < 5% of total range
        total_range = bic_scores[0] - min(bic_scores)
        lowest_n = len(bic_scores)
        for k in range(1, len(bic_scores)):
            improvement = bic_scores[k - 1] - bic_scores[k]
            if total_range > 0 and improvement / total_range < 0.05:
                lowest_n = k
                break
        lowest_n = max(2, min(lowest_n, 6))   # floor 2, cap 6

        self.bic_scores = bic_scores
        self.optimal_n  = lowest_n
        return lowest_n, bic_scores

    # ── GMM FIT ──────────────────────────────────────────────

    def fit_regime_models(self):
        gmm = GaussianMixture(
            n_components=self.optimal_n,
            covariance_type='full',
            random_state=42,
            n_init=10
        )
        gmm.fit(self.x_scaled)
        self.gmmModel      = gmm
        self.regimes       = gmm.predict(self.x_scaled)
        self.probabilities = gmm.predict_proba(self.x_scaled)
        return gmm, self.regimes, self.probabilities

    def fit_train_transform_test(self, df, train_ratio=0.7):
        split_idx  = int(len(df) * train_ratio)
        train_data = df.iloc[:split_idx]
        test_data  = df.iloc[split_idx:]

        scaler        = StandardScaler()
        X_train_sc    = scaler.fit_transform(train_data[self.regime_feat])
        X_test_sc     = scaler.transform(test_data[self.regime_feat])

        gmm = GaussianMixture(
            n_components=self.optimal_n,
            covariance_type='full',
            random_state=42,
            n_init=10
        )
        gmm.fit(X_train_sc)

        self.x_scaled      = np.vstack([X_train_sc, X_test_sc])
        self.scaler        = scaler
        self.gmmModel      = gmm
        self.regimes       = gmm.predict(self.x_scaled)
        self.probabilities = gmm.predict_proba(self.x_scaled)
        return self.regimes, self.probabilities

    # ── INTERPRET ────────────────────────────────────────────

    def interpret_regimes(self, df):
        df2 = df.copy()
        df2['regime'] = self.regimes
        regime_profiles = df2.groupby('regime')[self.regime_feat].agg(['mean', 'std'])
        self.regime_prof = regime_profiles
        print(regime_profiles)
        return regime_profiles

    # ── ALIGNMENT ────────────────────────────────────────────

    def align_regime_labels(self):
        """Sort regime labels by ascending mean Volatility_Regime for consistency."""
        if self.regime_prof is None:
            return
        vols       = self.regime_prof[('Volatility_Regime', 'mean')].values
        sorted_idx = np.argsort(vols)
        mapping    = {old: new for new, old in enumerate(sorted_idx)}
        self.regimes = np.array([mapping[r] for r in self.regimes])

    # ── DIAGNOSTICS ──────────────────────────────────────────

    def analyze_persistence(self):
        if self.regimes is None:
            print("ERROR: run regime detection first"); return
        changes     = np.sum(self.regimes[1:] != self.regimes[:-1])
        switch_rate = changes / len(self.regimes)
        self.regime_changes = changes
        self.switch_rate    = switch_rate
        tag = ("Excellent" if switch_rate < 0.10
               else "Good" if switch_rate < 0.20 else "Poor")
        print(f"{tag} Persistence  (switch_rate={switch_rate:.3f})")
        return switch_rate

    def analyze_confidence(self):
        if self.probabilities is None:
            print("ERROR: run regime detection first"); return
        avg_prob = self.probabilities.max(axis=1).mean()
        self.avg_confidence = avg_prob
        tag = ("Excellent" if avg_prob > 0.90
               else "Good" if avg_prob > 0.75
               else "Moderate" if avg_prob > 0.60 else "Heavy overlap")
        print(f"Average Confidence: {avg_prob:.3f}  ({tag})")

    def plot_bic(self):
        plt.plot(range(1, len(self.bic_scores) + 1), self.bic_scores, marker="o")
        plt.title(f"BIC — Optimal Regimes = {self.optimal_n}")
        plt.xlabel("Number of Regimes"); plt.ylabel("BIC Score")
        plt.show()

    def plot_regimes(self):
        price = self.instrument.indicator_data["Close"].loc[self.feature_data.index]
        plt.figure(figsize=(14, 6))
        sc = plt.scatter(price.index, price, c=self.regimes, cmap="tab10", s=10)
        plt.plot(price.index, price, alpha=0.3)
        plt.title(f"{self.instrument.symbol} Regime Detection")
        plt.colorbar(sc, label="Regime")
        plt.show()

    # ── RUN (full pipeline) ───────────────────────────────────

    def run(self, out_of_sample=True):
        self.instrument.fetch_data(self.period, self.interval)
        self.instrument.calculate_all_indicators()
        df = MetaModelDataset(self.instrument).create_features()
        self.feature_data = df

        if out_of_sample:
            # BIC on train half only
            split = int(len(df) * 0.7)
            scaler_tmp = StandardScaler()
            self.x_scaled = scaler_tmp.fit_transform(df.iloc[:split][self.regime_feat])
            self.find_optimal_regimes()
            self.fit_train_transform_test(df)
        else:
            self.prepare_regime_features(df)
            self.find_optimal_regimes()
            self.fit_regime_models()

        self.interpret_regimes(df)

    # ── STRATEGY BY REGIME (train-period) ────────────────────

    def strategy_by_regime(self):
        """
        Compute per-strategy returns on TRAIN data using regime-exit logic,
        then build a Sharpe-weighted regime→strategy mapping.
        """
        regime_df = self.feature_data.copy()
        regime_df["regime"] = self.regimes

        split_idx  = int(len(regime_df) * 0.7)
        train_df   = regime_df.iloc[:split_idx].copy()
        test_df    = regime_df.iloc[split_idx:].copy()
        self.train_df = train_df
        self.test_df  = test_df

        strategies = {
            "MovingAvg": MovingAverageCrossover(self.instrument, fast_period=5, slow_period=20),
            "Momentum":  Momentum(self.instrument),
            "MeanRev":   MeanReversion(self.instrument),
            "Breakout":  Breakout(self.instrument),
            "TrendFollow": TrendFollow(self.instrument, fast=20, slow=50),
        }

        close_full  = self.instrument.indicator_data["Close"].loc[regime_df.index]
        open_full   = self.instrument.indicator_data["Open"].loc[regime_df.index]
        slip_series = compute_slippage_series(close_full)

        # Build regime_series aligned to regime_df index
        regime_series_full = pd.Series(self.regimes, index=regime_df.index)

        strategy_names = list(strategies.keys())

        # ── Compute returns for all bars (train+test) with regime-exit ──
        for name, strat in strategies.items():
            sig = strat.generate_signals()["Signal"].shift(1).fillna(0)
            sig = np.clip(sig, -1, 1).reindex(regime_df.index).fillna(0)

            full_idx = regime_df.index
            ret_series = regime_exit_returns(
                signal_series   = sig,
                close_series    = close_full,
                open_series     = open_full,
                regime_series   = regime_series_full,
                probabilities   = self.probabilities,
                full_index      = full_idx,
                start_pos       = 0,
                end_pos         = len(full_idx) - 1,
                slippage_series = slip_series.reindex(full_idx).ffill(),
                strategy_name   = name,
            )
            regime_df[name] = ret_series.values

        self.strategy_returns_df = regime_df[strategy_names].copy()
        train_df = regime_df.iloc[:split_idx].copy()
        test_df  = regime_df.iloc[split_idx:].copy()
        self.train_returns_df = train_df[strategy_names]
        self.test_returns_df  = test_df[strategy_names]

        # ── Build metrics on TRAIN only ──────────────────────
        results = []
        for regime in sorted(train_df["regime"].unique()):
            subset = train_df[train_df["regime"] == regime]
            for strat_name in strategy_names:
                rets = subset[strat_name].dropna()
                if len(rets) == 0:
                    continue
                mu  = rets.mean()
                vol = rets.std()
                sharpe = mu / vol if vol != 0 else 0
                equity = (1 + rets).cumprod()
                dd     = ((equity - equity.cummax()) / equity.cummax()).min()
                results.append({
                    "Regime":      regime,
                    "Strategy":    strat_name,
                    "Mean_Return": mu,
                    "Win_Rate":    (rets > 0).mean(),
                    "Sharpe":      sharpe,
                    "Volatility":  vol,
                    "Max_Drawdown": dd,
                })

        metrics_df = pd.DataFrame(results)
        self.strategy_regime_results = metrics_df

        # Best strategy per regime
        mapping = {}
        for regime in metrics_df["Regime"].unique():
            best = (metrics_df[metrics_df["Regime"] == regime]
                    .sort_values("Sharpe", ascending=False).iloc[0])
            mapping[regime] = {
                "Strategy":    best["Strategy"],
                "Sharpe":      best["Sharpe"],
                "Win_Rate":    best["Win_Rate"],
                "Mean_Return": best["Mean_Return"],
            }
        self.strategy_regime_max = mapping
        return metrics_df, pd.DataFrame(mapping)

