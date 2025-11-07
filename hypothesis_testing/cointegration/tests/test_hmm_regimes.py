import numpy as np
import pandas as pd
from pathlib import Path

from hypothesis_testing.cointegration.hmm_regimes import (
    build_symbol_features,
    fit_hmms_mvp,
    label_symbols,
)


def _make_synthetic_price(T: int = 1000, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    # Two regimes: low vol then high vol
    r_low = rng.normal(0.0, 0.002, size=T // 2)
    r_high = rng.normal(0.0, 0.01, size=T - T // 2)
    r = np.concatenate([r_low, r_high])
    p = 100.0 * np.exp(np.cumsum(r))
    idx = pd.date_range("2024-01-01", periods=T, freq="15min")
    return pd.Series(p, index=idx, name="AAA_close")


def test_hmm_labels_two_regimes():
    series = _make_synthetic_price(T=2000)
    price_df = series.to_frame()
    feats = build_symbol_features(price_df, bars_per_day=96)
    assert "AAA" in feats and not feats["AAA"].empty

    models = fit_hmms_mvp(feats, random_state=0, n_iter=50)
    assert "AAA" in models

    labels = label_symbols(models, feats)
    assert f"AAA_hmm_state" in labels.columns

    # Both states present
    counts = labels[f"AAA_hmm_state"].value_counts()
    assert len(counts) == 2


