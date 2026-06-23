
def align_regime_labels_hungarian(train_reg, test_regimes, test_probabilities,
                                  test_feature_data):
    """
    Align test GMM regime labels to train regime labels using Hungarian algorithm
    on feature-mean correlation.  Returns remapped (test_regimes, test_probabilities).
    """
    if train_reg.regime_prof is None:
        print("WARNING: Missing train regime profiles — skipping alignment")
        return test_regimes, test_probabilities

    # Compute test regime profiles
    test_df = test_feature_data.copy()
    test_df["regime"] = test_regimes
    test_prof = test_df.groupby("regime")[train_reg.regime_feat].mean()

    train_prof = train_reg.regime_prof.xs("mean", axis=1, level=1)

    n_train = len(train_prof)
    n_test  = len(test_prof)
    n       = min(n_train, n_test)

    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr = np.corrcoef(train_prof.iloc[i].values,
                               test_prof.iloc[j].values)[0, 1]
            cost[i, j] = -corr if not np.isnan(corr) else 1.0

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {col_ind[j]: row_ind[j] for j in range(len(col_ind))}
    print(f"Regime alignment mapping: {mapping}")

    new_regimes = np.array([mapping.get(r, r) for r in test_regimes])

    # Remap probability columns
    k = test_probabilities.shape[1]
    new_probs = np.zeros_like(test_probabilities)
    for old_id, new_id in mapping.items():
        if old_id < k and new_id < k:
            new_probs[:, new_id] = test_probabilities[:, old_id]

    return new_regimes, new_probs

