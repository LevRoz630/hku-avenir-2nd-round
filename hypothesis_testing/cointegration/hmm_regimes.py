"""
Hidden Markov Model (HMM) regimes for per-symbol labeling and masking.

MVP implementation:
- Features per symbol: 15m log-return r_t and 1d rolling volatility (std of r_t over bars_per_day)
- Standardize features per symbol (fit on cointegration_data)
- Fit GaussianHMM with n_states=2, diag covariance
- Label states, ordered by mean volatility: {low, high}
- Persist/load labels (Parquet) with minimal metadata
- Basket-wise regime filter utility (policy='all')
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


@dataclass
class SymbolHMM:
    symbol: str
    scaler: StandardScaler
    model: GaussianHMM
    feature_columns: List[str]
    state_ordering: List[int]  # mapping from raw state id -> ordered index (low->high vol)


def _extract_symbol_from_col(col: str) -> str:
    # Columns are expected as {symbol}_close
    if col.endswith("_close"):
        return col[: -len("_close")]
    return col


def build_symbol_features(price_data: pd.DataFrame, bars_per_day: int) -> Dict[str, pd.DataFrame]:
    """
    Build per-symbol feature matrices on a shared index:
    - r_t: log return
    - vol_1d: rolling std of r_t over bars_per_day

    Returns dict[symbol] -> DataFrame with columns ['r_t', 'vol_1d']
    """
    if not isinstance(price_data.index, pd.DatetimeIndex):
        raise ValueError("price_data must have DatetimeIndex")
    if bars_per_day <= 0:
        raise ValueError("bars_per_day must be positive")

    features_by_symbol: Dict[str, pd.DataFrame] = {}

    for col in price_data.columns:
        symbol = _extract_symbol_from_col(col)
        series = price_data[col].astype(float)
        # Require strictly positive prices for log returns
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        series = series[series > 0]
        if series.empty:
            continue

        r_t = np.log(series / series.shift(1))
        vol_1d = r_t.rolling(window=bars_per_day, min_periods=bars_per_day // 2).std(ddof=1)

        df = pd.DataFrame({"r_t": r_t, "vol_1d": vol_1d}).dropna()
        if len(df) < max(100, 10):  # need a reasonable minimum
            continue
        features_by_symbol[symbol] = df

    # Align features on intersection of indices per symbol (not necessary to force global align)
    return features_by_symbol


def fit_hmms_mvp(
    features_by_symbol: Dict[str, pd.DataFrame],
    *,
    random_state: int = 0,
    n_iter: int = 200,
) -> Dict[str, SymbolHMM]:
    """
    Fit per-symbol 2-state GaussianHMM with diag covariance on standardized features.
    Enforce minimal quality: each state ≥10% support and avg dwell time ≥24 bars.
    Symbols failing checks are skipped.
    """
    models: Dict[str, SymbolHMM] = {}
    for symbol, feats in features_by_symbol.items():
        # Standardize features per symbol
        scaler = StandardScaler()
        X = scaler.fit_transform(feats.values)

        # Train HMM
        hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=n_iter, random_state=random_state)
        try:
            hmm.fit(X)
        except Exception:
            continue

        # Viterbi decode to assess supports and dwell time
        states = hmm.predict(X)
        # State ordering by mean volatility (use original vol_1d column as proxy)
        state_means = []
        for s in range(hmm.n_components):
            mask = states == s
            support = mask.mean() if len(mask) else 0.0
            mean_vol = float(feats.loc[mask, "vol_1d"].mean()) if mask.any() else np.nan
            state_means.append((s, support, mean_vol))

        # Quality checks
        supports = [sup for _, sup, _ in state_means if not np.isnan(sup)]
        if len(supports) != hmm.n_components or min(supports) < 0.10:
            continue

        # Avg dwell time estimation
        avg_dwell = _estimate_avg_dwell(states)
        if avg_dwell < 24:  # at least ~6 hours on 15m bars
            continue

        # Order states by mean volatility ascending: low(0) -> high(1)
        ordering = [x[0] for x in sorted(state_means, key=lambda t: (np.inf if np.isnan(t[2]) else t[2]))]

        models[symbol] = SymbolHMM(
            symbol=symbol,
            scaler=scaler,
            model=hmm,
            feature_columns=list(feats.columns),
            state_ordering=ordering,
        )
    return models


def _estimate_avg_dwell(states: np.ndarray) -> float:
    if states.size == 0:
        return 0.0
    runs = 1
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            runs += 1
    return float(len(states) / runs) if runs > 0 else float(len(states))


def label_symbols(
    models: Dict[str, SymbolHMM],
    features_by_symbol: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Predict per-symbol regimes and return wide DataFrame with columns `{SYMBOL}_hmm_state`.
    States are remapped to volatility-ordered labels {0:low, 1:high}.
    Index is the intersection between feature indices used for each symbol.
    """
    label_frames: List[pd.DataFrame] = []
    common_index: Optional[pd.DatetimeIndex] = None

    # Determine common index across available symbols
    for symbol, model in models.items():
        feats = features_by_symbol.get(symbol)
        if feats is None or feats.empty:
            continue
        common_index = feats.index if common_index is None else common_index.intersection(feats.index)

    if common_index is None or len(common_index) == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([], name=None))

    for symbol, model in models.items():
        feats = features_by_symbol.get(symbol)
        if feats is None or feats.empty:
            continue
        X = model.scaler.transform(feats.loc[common_index, model.feature_columns].values)
        raw_states = model.model.predict(X)
        # Remap to ordered states
        remap = {raw: i for i, raw in enumerate(model.state_ordering)}
        mapped = np.array([remap.get(s, s) for s in raw_states], dtype=int)
        label_frames.append(pd.DataFrame({f"{symbol}_hmm_state": mapped}, index=common_index))

    if not label_frames:
        return pd.DataFrame(index=common_index)
    labels = pd.concat(label_frames, axis=1)
    return labels


def persist_labels(labels_df: pd.DataFrame, path: Path, meta: Optional[Dict[str, object]] = None) -> None:
    """Persist labels to Parquet and optional metadata to adjacent JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(path)
    if meta:
        import json
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, default=str)


def load_labels(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def apply_regime_filter(
    price_data: pd.DataFrame,
    labels_df: pd.DataFrame,
    basket: Sequence[str],
    include_states: Optional[Set[int]] = None,
    policy: str = "all",
    k: Optional[int] = None,
) -> pd.DataFrame:
    """
    Filter price_data rows by per-symbol regimes for the given basket.

    MVP supports policy='all' only: keep rows where all basket symbols are in include_states.
    """
    if include_states is None:
        return price_data

    # Align on common index
    idx = price_data.index
    if not labels_df.index.equals(idx):
        idx = idx.intersection(labels_df.index)
    if len(idx) == 0:
        return price_data.iloc[0:0]

    basket_cols = [f"{s}_close" for s in basket]
    state_cols = [f"{s}_hmm_state" for s in basket]

    missing_states = [c for c in state_cols if c not in labels_df.columns]
    if missing_states:
        # If any symbol lacks regimes, return empty to avoid leakage
        return price_data.loc[idx].iloc[0:0]

    states = labels_df.loc[idx, state_cols]
    if policy != "all":
        raise ValueError("Only policy='all' is supported in MVP")

    mask = np.ones(len(idx), dtype=bool)
    for col in state_cols:
        mask &= states[col].isin(include_states).values

    filtered_idx = idx[mask]
    if len(filtered_idx) == 0:
        return price_data.loc[idx].iloc[0:0]
    # Return rows only for timestamps present after masking
    return price_data.loc[filtered_idx, [c for c in price_data.columns if c in basket_cols or not c.endswith("_close") or True]]


def train_and_persist_labels(
    cointegration_data: pd.DataFrame,
    bars_per_day: int,
    output_path: Path,
    meta: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """
    Convenience orchestration: build features, fit HMMs, label symbols, persist labels.
    Returns the labels DataFrame.
    """
    feats = build_symbol_features(cointegration_data, bars_per_day)
    models = fit_hmms_mvp(feats)
    labels = label_symbols(models, feats)
    persist_labels(labels, output_path, meta=meta)
    return labels



