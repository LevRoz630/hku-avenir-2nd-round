"""
Johansen cointegration test implementation.
"""

import numpy as np
from typing import Dict
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy.stats import chi2


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
    result = coint_johansen(log_prices, det_order, k_ar_diff)
    
    # Trace statistic for rank 0 hypothesis
    trace_stat = result.lr1[0]  # Test H0: rank <= 0 vs H1: rank > 0
    trace_crit = result.cvt[0, 1]  # 5% critical value
    
    # P-value approximation (chi-square distribution)
    # Degrees of freedom for trace test
    n = log_prices.shape[1]
    df = n
    p_value = 1 - chi2.cdf(trace_stat, df)
    
    return {
        'trace_stat': trace_stat,
        'trace_crit': trace_crit,
        'p_value': p_value,
        'eigenvalues': result.eig,
        'eigenvectors': result.evec,
        'is_cointegrated': p_value < p_value_threshold
    }

