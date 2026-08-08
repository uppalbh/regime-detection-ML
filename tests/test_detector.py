"""GMM detector unit tests on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from regime_ml.config import REGIME_FEATURES
from regime_ml.data.loader import Instrument
from regime_ml.features.regime_features import build_regime_features
from regime_ml.regimes.detector import RegimeDetector


def _instrument(n: int = 300) -> Instrument:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-01-01", periods=n)
    # Two-regime synthetic process
    rets = np.where(
        np.arange(n) < n // 2,
        rng.normal(0.001, 0.005, n),
        rng.normal(-0.0005, 0.02, n),
    )
    close = 100 * np.cumprod(1 + rets)
    data = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1e6,
        },
        index=idx,
    )
    inst = Instrument("SYN")
    inst.data = data
    inst.period = "synthetic"
    inst.interval = "1d"
    return inst


def test_detector_selects_k_and_fits():
    inst = _instrument()
    feats = build_regime_features(inst.data)
    inst.features = feats
    reg = RegimeDetector(inst, "synthetic", "1d")
    train = feats.iloc[: int(len(feats) * 0.7)]
    reg.feature_data = train.copy()
    scaler = StandardScaler()
    x = scaler.fit_transform(train[REGIME_FEATURES])
    reg.scaler = scaler
    k = reg.find_optimal_k(x)
    assert 2 <= k <= 6
    reg.fit_gmm(x)
    assert reg.regimes is not None
    assert reg.probabilities.shape == (len(train), k)
    reg.interpret_regimes()
    assert reg.regime_prof is not None
