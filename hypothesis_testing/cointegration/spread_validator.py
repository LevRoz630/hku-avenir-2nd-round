"""
Spread widening validation and z-score computation.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_max_zscore(spread: np.ndarray, lookback_days: int = 90) -> float:
    """
    Compute maximum absolute z-score of spread over rolling lookback window.
    Optimized using vectorized pandas operations.
    
    Parameters:
    -----------
    spread : np.ndarray, shape (T,)
        Spread series
    lookback_days : int
        Lookback window size in days
        
    Returns:
    --------
    float
        Maximum absolute z-score
    """
    import pandas as pd
    
    lookback_bars = lookback_days * 96  # 96 bars per day (15-minute bars)
    T = len(spread)
    
    if T < lookback_bars:
        return np.inf
    
    # Convert to pandas Series for vectorized rolling operations
    spread_series = pd.Series(spread)
    
    # Compute rolling mean and std (much faster than Python loop)
    rolling_mean = spread_series.rolling(window=lookback_bars, min_periods=lookback_bars).mean()
    rolling_std = spread_series.rolling(window=lookback_bars, min_periods=lookback_bars).std()
    
    # Compute z-scores vectorized
    z_scores = (spread_series - rolling_mean) / rolling_std
    z_scores = z_scores.iloc[lookback_bars:]  # Skip first lookback_bars NaN values
    
    # Return max absolute z-score
    if len(z_scores) == 0 or z_scores.isna().all():
        return np.inf
    
    max_z = z_scores.abs().max()
    return float(max_z) if not np.isnan(max_z) else np.inf


