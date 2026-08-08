"""Data and regime-feature tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_ml.config import REGIME_FEATURES
from regime_ml.data.loader import DataValidationError, clean_ohlcv
from regime_ml.features.regime_features import build_regime_features


def _ohlcv(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1e5, 1e6, n),
        },
        index=idx,
    )


def test_clean_sorts_and_dedups():
    df = _ohlcv(20)
    messy = pd.concat([df.iloc[[3]], df]).sort_index(ascending=False)
    cleaned, report = clean_ohlcv(messy)
    assert cleaned.index.is_monotonic_increasing
    assert report["final_rows"] == 20


def test_clean_rejects_empty():
    with pytest.raises(DataValidationError):
        clean_ohlcv(pd.DataFrame())


def test_regime_features_exist_and_finite():
    feats = build_regime_features(_ohlcv(150))
    for col in REGIME_FEATURES:
        assert col in feats.columns
        assert np.isfinite(feats[col].to_numpy()).all()


def test_regime_features_are_causal():
    df = _ohlcv(160)
    full = build_regime_features(df)
    # Align on common index after dropna warmup
    cutoff_date = full.index[40]
    truncated = build_regime_features(df.loc[:cutoff_date])
    for col in REGIME_FEATURES:
        assert np.isclose(full.loc[cutoff_date, col], truncated.loc[cutoff_date, col])
