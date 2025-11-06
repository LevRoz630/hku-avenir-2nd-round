"""
Johansen cointegration test implementation.
"""

import numpy as np
from typing import Dict
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy.stats import chi2
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
    k_ar_diff : int
        Number of lags in the VAR model
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
    trace_crit = result.cvt[0, 1]  # 5% critical value
    
    # Validate trace_stat
    if np.isnan(trace_stat) or np.isinf(trace_stat):
        raise ValueError(f"Invalid trace statistic: {trace_stat}")
    
    # P-value approximation (chi-square distribution)
    # Degrees of freedom for trace test: n (number of assets)
    df = n
    p_value = 1 - chi2.cdf(trace_stat, df)
    
    # Ensure p_value is valid
    if np.isnan(p_value) or np.isinf(p_value):
        p_value = 1.0  # Conservative: assume no cointegration if p-value is invalid
    
    return {
        'trace_stat': float(trace_stat),
        'trace_crit': float(trace_crit),
        'p_value': float(p_value),
        'eigenvalues': result.eig.tolist(),
        'eigenvectors': result.evec.tolist(),
        'is_cointegrated': p_value < p_value_threshold
    }

