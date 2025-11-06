"""
Persistence testing for cointegration over rolling windows.
"""

import numpy as np
from typing import Dict
import logging

from .johansen_test import johansen_test

logger = logging.getLogger(__name__)


def test_persistence_rolling(log_prices: np.ndarray, eigenvector: np.ndarray,
                            window_days: int = 90, step_days: int = 30,
                            p_value_threshold: float = 0.01) -> float:
    """
    Test cointegration persistence over rolling windows.
    Returns persistence ratio (fraction of windows with p < threshold).
    
    Parameters:
    -----------
    log_prices : np.ndarray, shape (T, n)
        Log prices for n assets over T time periods
    eigenvector : np.ndarray, shape (n,)
        Cointegration eigenvector (not used directly, but kept for consistency)
    window_days : int
        Size of rolling window in days
    step_days : int
        Step size between windows in days
    p_value_threshold : float
        P-value threshold for cointegration (default: 0.01 = 1%)
        
    Returns:
    --------
    float
        Persistence ratio (0.0 to 1.0)
    """
    # Convert days to 15-minute bars (96 bars per day)
    window_bars = window_days * 96
    step_bars = step_days * 96
    
    T, n = log_prices.shape
    if T < window_bars:
        return 0.0
    
    p_values = []
    for start in range(0, T - window_bars + 1, step_bars):
        end = start + window_bars
        window_data = log_prices[start:end]
        
        try:
            result = johansen_test(window_data, p_value_threshold=p_value_threshold)
            p_values.append(result['p_value'])
        except:
            p_values.append(1.0)  # Fail conservatively
    
    if not p_values:
        return 0.0
    
    persistence_ratio = sum(p < p_value_threshold for p in p_values) / len(p_values)
    return persistence_ratio

