
class RegimeBacktester:

    def __init__(self, regime_detector):
        self.regime_detector = regime_detector
        self.strategy_returns      = None   # pd.DataFrame (dates × strategies)
        self.regime_weights        = None
        self.daily_strategy_weights = None
        self.portfolio_returns     = None
        self.equity_curve          = None
        self.static_results        = None
        self.equal_weight_results  = None
        self.adaptive_results      = None

    def build_regime_weight(self):
        """
        Build a dict  {regime_id: {strategy: sharpe_weight}}
        using train-only Sharpe from strategy_regime_results.
        Called internally; if strategy_returns already set (walk-forward),
        skip calling strategy_by_regime again.
        """
        rd = self.regime_detector

        # In walk-forward the wrapper already has strategy_regime_results
        if rd.strategy_regime_results is None:
            rd.strategy_by_regime()

        if self.strategy_returns is None:
            self.strategy_returns = rd.strategy_returns_df

        metrics_df = rd.strategy_regime_results
        strategies = ["MovingAvg", "Momentum", "MeanRev", "Breakout", "TrendFollow"]
        weights_df = {}

        for i in range(rd.optimal_n):
            regime_data = metrics_df[metrics_df["Regime"] == i]
            pos_sharpes = {}
            for s in strategies:
                row = regime_data[regime_data["Strategy"] == s]
                if len(row) > 0:
                    sh = row["Sharpe"].iloc[0]
                    if sh > 0:
                        pos_sharpes[s] = sh

            total = sum(pos_sharpes.values())
            weights_df[i] = {
                s: (pos_sharpes[s] / total if s in pos_sharpes and total > 0 else 0)
                for s in strategies
            }

        self.regime_weights = weights_df
        return weights_df

    def build_daily_strategy_weights(self):
        if self.regime_weights is None:
            self.build_regime_weight()

        probs      = self.regime_detector.probabilities
        strategies = ["MovingAvg", "Momentum", "MeanRev", "Breakout", "TrendFollow"]
        daily      = []

        for day_probs in probs:
            w = {}
            for s in strategies:
                w[s] = sum(
                    day_probs[r] * self.regime_weights[r][s]
                    for r in range(len(day_probs))
                )
            daily.append(w)

        self.daily_strategy_weights = pd.DataFrame(
            daily, index=self.strategy_returns.index
        )
        return self.daily_strategy_weights

    def run_adaptive_portfolio(self):
        if self.daily_strategy_weights is None:
            self.build_daily_strategy_weights()
        self.portfolio_returns = (
            self.daily_strategy_weights * self.strategy_returns
        ).sum(axis=1)
        self.equity_curve    = (1 + self.portfolio_returns).cumprod()
        self.adaptive_results = self.evaluate_portfolio(self.portfolio_returns)
        return self.portfolio_returns

    def evaluate_portfolio(self, returns):
        total_ret   = (1 + returns).prod() - 1
        n           = max(len(returns), 1)
        annual_ret  = (1 + total_ret) ** (252 / n) - 1
        annual_vol  = returns.std() * np.sqrt(252)
        sharpe      = annual_ret / annual_vol if annual_vol != 0 else 0
        equity      = (1 + returns).cumprod()
        peak        = equity.cummax()
        max_dd      = ((equity - peak) / peak).min()
        win_rate    = (returns > 0).mean()
        return {
            "Total_Return":    total_ret,
            "Annual_Return":   annual_ret,
            "Annual_Volatility": annual_vol,
            "Sharpe":          sharpe,
            "Max_Drawdown":    max_dd,
            "Win_Rate":        win_rate,
        }

    def run_static_portfolio(self):
        results = []
        for s in ["MovingAvg", "Momentum", "MeanRev", "Breakout", "TrendFollow"]:
            m = self.evaluate_portfolio(self.strategy_returns[s])
            m["Strategy"] = s
            results.append(m)
        self.static_results = pd.DataFrame(results)
        return self.static_results

    def run_equal_weight_portfolio(self):
        rets = self.strategy_returns.mean(axis=1)
        self.equal_weight_results = self.evaluate_portfolio(rets)
        return self.equal_weight_results

    def compare_strategies(self):
        if self.static_results        is None: self.run_static_portfolio()
        if self.adaptive_results      is None: self.run_adaptive_portfolio()
        if self.equal_weight_results  is None: self.run_equal_weight_portfolio()

        keys = set(self.equal_weight_results) | set(self.adaptive_results)
        static_row = {
            "Total_Return":    self.static_results["Total_Return"].mean(),
            "Annual_Return":   self.static_results["Annual_Return"].mean(),
            "Annual_Volatility": self.static_results["Annual_Volatility"].mean(),
            "Sharpe":          self.static_results["Sharpe"].mean(),
            "Max_Drawdown":    self.static_results["Max_Drawdown"].min(),
            "Win_Rate":        self.static_results["Win_Rate"].mean(),
        }
        return pd.DataFrame(
            [static_row,
             {k: self.equal_weight_results.get(k, 0) for k in keys},
             {k: self.adaptive_results.get(k, 0)     for k in keys}],
            index=["Static", "EqualWeight", "Adaptive"]
        )

    def sharpe_significance_test(self):
        if self.portfolio_returns is None:
            print("Run adaptive portfolio first"); return None, None
        adaptive = self.portfolio_returns.dropna()
        # Use the best static strategy, not just MA
        best_static = self.static_results.sort_values("Sharpe", ascending=False).iloc[0]["Strategy"]
        static = self.strategy_returns[best_static].loc[adaptive.index].dropna()
        n        = min(len(adaptive), len(static))
        if n < 2:
            print("Insufficient data"); return None, None
        t, p = ttest_rel(adaptive.iloc[:n], static.iloc[:n])
        print(f"T-stat: {t:.3f}   P-value: {p:.4f}")
        return t, p

