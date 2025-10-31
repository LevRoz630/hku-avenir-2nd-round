"""
Spread widening validation and z-score computation.
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def compute_max_zscore(spread: np.ndarray, lookback_days: int = 90) -> float:
    """
    Compute maximum absolute z-score of spread over rolling lookback window.
    
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
    lookback_bars = lookback_days * 96  # 96 bars per day (15-minute bars)
    T = len(spread)
    
    if T < lookback_bars:
        return np.inf
    
    max_z = 0.0
    for i in range(lookback_bars, T):
        window_spread = spread[i - lookback_bars:i]
        mean = np.mean(window_spread)
        std = np.std(window_spread)
        
        if std > 0:
            z = abs((spread[i] - mean) / std)
            max_z = max(max_z, z)
    
    return max_z


def validate_spread_constraints(spread: np.ndarray, max_zscore_threshold: float = 3.0,
                                lookback_days: int = 90) -> Dict:
    """
    Validate spread constraints and return metrics.
    
    Parameters:
    -----------
    spread : np.ndarray
        Spread series
    max_zscore_threshold : float
        Maximum allowed z-score (default 3.0)
    lookback_days : int
        Lookback window for z-score computation
        
    Returns:
    --------
    Dict
        Dictionary with 'max_zscore', 'is_valid', and other metrics
    """
    max_z = compute_max_zscore(spread, lookback_days)
    
    return {
        'max_zscore': max_z,
        'is_valid': max_z < max_zscore_threshold,
        'mean': np.mean(spread),
        'std': np.std(spread)
    }

