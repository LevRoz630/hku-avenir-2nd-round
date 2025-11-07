"""
Johansen cointegration test implementation.
"""

import numpy as np
from typing import Dict
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import logging

logger = logging.getLogger(__name__)


def johansen_test(log_prices: np.ndarray, det_order: int = -1, k_ar_diff: int = 1, 
                  p_value_threshold: float = 0.01) -> Dict:
    """
    Perform Johansen cointegration test.
    
    Parameters:
    -----------
    log_prices : np.ndarray, shape (T, n)
        Log prices for n assets over T time periods
    det_order : int
        Deterministic order: -1 (no deterministic), 0 (constant), 1 (linear trend)
        For cointegration testing, -1 (no deterministic) is most appropriate
    k_ar_diff : int
        Number of lags in the VAR model. Default 1 is reasonable for most applications
    p_value_threshold : float
        P-value threshold for cointegration (default: 0.01 = 1%)
        
    Returns:
    --------
    dict with keys: 'trace_stat', 'trace_crit', 'p_value', 'eigenvalues', 'eigenvectors', 'is_cointegrated'
    """
    # Input validation
    if not isinstance(log_prices, np.ndarray):
        raise ValueError("log_prices must be a numpy array")
    
    if log_prices.ndim != 2:
        raise ValueError(f"log_prices must be 2D array, got shape {log_prices.shape}")
    
    T, n = log_prices.shape
    
    if T < n * 10:
        raise ValueError(f"Insufficient observations: T={T} < n*10={n*10}. Need at least 10 observations per asset.")
    
    if n < 2:
        raise ValueError(f"Need at least 2 assets for cointegration test, got n={n}")
    
    # Check for NaN/Inf values
    if np.any(np.isnan(log_prices)) or np.any(np.isinf(log_prices)):
        raise ValueError("log_prices contains NaN or Inf values")
    
    # Validate det_order
    if det_order not in [-1, 0, 1]:
        raise ValueError(f"det_order must be -1, 0, or 1, got {det_order}")
    
    # Validate k_ar_diff
    if k_ar_diff < 0:
        raise ValueError(f"k_ar_diff must be >= 0, got {k_ar_diff}")
    
    # Validate p_value_threshold
    if not 0 < p_value_threshold <= 1:
        raise ValueError(f"p_value_threshold must be in (0, 1], got {p_value_threshold}")
    
    try:
        result = coint_johansen(log_prices, det_order, k_ar_diff)
    except Exception as e:
        logger.error(f"Johansen test failed: {e}")
        raise ValueError(f"Johansen test computation failed: {e}")
    
    # Trace statistic for rank 0 hypothesis
    trace_stat = result.lr1[0]  # Test H0: rank <= 0 vs H1: rank > 0
    
    # Validate trace_stat
    if np.isnan(trace_stat) or np.isinf(trace_stat):
        raise ValueError(f"Invalid trace statistic: {trace_stat}")
    
    # Use critical values directly instead of chi-square approximation
    # Johansen test follows non-standard distribution, so we compare against critical values
    # cvt[0, 0] = 1% critical value, cvt[0, 1] = 5% critical value
    trace_crit_1pct = result.cvt[0, 0]  # 1% critical value
    trace_crit_5pct = result.cvt[0, 1]  # 5% critical value (kept for reference)
    
    # Determine p-value threshold level and use appropriate critical value
    if p_value_threshold <= 0.01:
        trace_crit = trace_crit_1pct
        p_value_approx = 0.01 if trace_stat > trace_crit_1pct else 0.05
    elif p_value_threshold <= 0.05:
        trace_crit = trace_crit_5pct
        p_value_approx = 0.05 if trace_stat > trace_crit_5pct else 0.10
    else:
        trace_crit = trace_crit_5pct
        p_value_approx = 0.10
    
    # Compare trace statistic directly against critical value
    is_cointegrated = trace_stat > trace_crit
    
    return {
        'trace_stat': float(trace_stat),
        'trace_crit': float(trace_crit),
        'trace_crit_1pct': float(trace_crit_1pct),
        'trace_crit_5pct': float(trace_crit_5pct),
        'p_value': float(p_value_approx),  # Approximate p-value based on critical value comparison
        'eigenvalues': result.eig.tolist(),
        'eigenvectors': result.evec.tolist(),
        'is_cointegrated': is_cointegrated
    }

