"""
Filter baskets by mean reversion speed.
Computes half-life and ADF test metrics with multiprocessing.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import os
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


def _compute_half_life(spread: np.ndarray) -> float:
    """
    Compute half-life of mean reversion using Ornstein-Uhlenbeck process.
    Estimates theta from: spread_t = theta * spread_{t-1} + epsilon_t
    
    Half-life = -log(2) / log(theta) if theta < 1
    
    Parameters:
    -----------
    spread : np.ndarray, shape (T,)
        Spread series
        
    Returns:
    --------
    float
        Half-life in periods (or np.inf if non-stationary)
    """
    if len(spread) < 2:
        return np.inf
    
    # De-mean the spread
    spread_demeaned = spread - np.mean(spread)
    
    # Create lagged series
    spread_lag = spread_demeaned[:-1]
    spread_diff = np.diff(spread_demeaned)
    
    # OLS regression: spread_diff = theta * spread_lag + epsilon
    if np.var(spread_lag) == 0:
        return np.inf
    
    theta = np.dot(spread_lag, spread_diff) / np.dot(spread_lag, spread_lag)
    
    # Ensure theta is valid for mean reversion (0 < theta < 1)
    if theta <= 0 or theta >= 1:
        return np.inf
    
    # Half-life = -log(2) / log(theta)
    half_life = -np.log(2) / np.log(theta)
    
    return half_life


def _compute_adf_test(spread: np.ndarray) -> Dict:
    """
    Compute Augmented Dickey-Fuller test on spread.
    
    Parameters:
    -----------
    spread : np.ndarray, shape (T,)
        Spread series
        
    Returns:
    --------
    dict with keys: 'adf_statistic', 'p_value', 'is_stationary'
    """
    if len(spread) < 10:
        return {
            'adf_statistic': np.nan,
            'p_value': 1.0,
            'is_stationary': False
        }
    
    try:
        # ADF test with automatic lag selection
        result = adfuller(spread, autolag='AIC')
        
        adf_statistic = result[0]
        p_value = result[1]
        is_stationary = p_value < 0.01  # Strong stationarity signal
        
        return {
            'adf_statistic': adf_statistic,
            'p_value': p_value,
            'is_stationary': is_stationary
        }
    except Exception as e:
        logger.debug(f"Error in ADF test: {e}")
        return {
            'adf_statistic': np.nan,
            'p_value': 1.0,
            'is_stationary': False
        }


def _test_basket_mean_reversion(args):
    """
    Worker function to compute mean reversion metrics for a single basket.
    Uses half_life_data to recompute spread (prevents data leakage).
    Designed to be picklable for multiprocessing.
    """
    basket_result, half_life_data_dict, half_life_threshold_days, bars_per_day = args
    
    try:
        # Reconstruct half_life_data DataFrame
        half_life_data = pd.DataFrame(
            half_life_data_dict['data'],
            index=pd.DatetimeIndex(half_life_data_dict['index']),
            columns=half_life_data_dict['columns']
        )
        
        # Extract basket prices from half_life_data
        basket = basket_result['basket']
        basket_cols = [f'{sym}_close' for sym in basket]
        
        if not all(col in half_life_data.columns for col in basket_cols):
            return None
        
        basket_prices = half_life_data[basket_cols].values
        
        # Convert to log prices
        log_prices = np.log(basket_prices)
        
        # Recompute spread using eigenvector from cointegration test
        eigenvector = np.array(basket_result['eigenvector'])
        spread = log_prices @ eigenvector
        
        # Compute half-life
        half_life_periods = _compute_half_life(spread)
        half_life_days = half_life_periods / bars_per_day
        
        # Compute ADF test
        adf_result = _compute_adf_test(spread)
        
        # Determine if basket passes filter
        # Pass if: half_life < threshold OR ADF p-value < 0.01
        passes = (half_life_days < half_life_threshold_days) or adf_result['is_stationary']
        
        return {
            'basket': basket_result['basket'],
            'half_life_periods': half_life_periods,
            'half_life_days': half_life_days,
            'adf_statistic': adf_result['adf_statistic'],
            'adf_p_value': adf_result['p_value'],
            'is_stationary': adf_result['is_stationary'],
            'passes_filter': passes
        }
    except Exception as e:
        logger.debug(f"Error testing basket {basket_result.get('basket', 'unknown')}: {e}")
        return None


def filter_baskets_mean_reversion(baskets: List[Dict],
                                  half_life_data: pd.DataFrame,
                                  half_life_threshold_days: float = 30.0,
                                  bars_per_day: int = 24,
                                  max_workers: Optional[int] = None) -> List[Dict]:
    """
    Filter baskets by mean reversion speed.
    Uses separate half_life_data to recompute spreads (prevents data leakage).
    
    Parameters:
    -----------
    baskets : List[Dict]
        List of basket dictionaries with 'eigenvector' key (from cointegration test)
    half_life_data : pd.DataFrame
        Price data for half-life testing (should be separate from cointegration data)
    half_life_threshold_days : float
        Maximum half-life in days to pass filter (default 30 days)
    bars_per_day : int
        Number of bars per day (24 for 1h, 96 for 15m)
    max_workers : Optional[int]
        Number of parallel workers. If None, uses CPU count.
        
    Returns:
    --------
    List[Dict]
        Filtered baskets with added 'mean_reversion' metrics
    """
    if max_workers is None:
        max_workers = int(0.9 * (os.cpu_count() or 1)) or 1
    
    logger.info(f"Testing {len(baskets)} baskets for mean reversion speed "
               f"(using separate half-life data: {len(half_life_data)} samples)...")
    
    # Convert DataFrame to dict for pickling
    half_life_data_dict = {
        'data': half_life_data.values,
        'index': half_life_data.index.values,
        'columns': half_life_data.columns.tolist()
    }
    
    # Prepare arguments for parallel processing
    args_list = [
        (basket_result, half_life_data_dict, half_life_threshold_days, bars_per_day)
        for basket_result in baskets
    ]
    
    # Test baskets in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_basket = {
            executor.submit(_test_basket_mean_reversion, args): args[0]['basket']
            for args in args_list
        }
        
        completed = 0
        for future in as_completed(future_to_basket):
            completed += 1
            try:
                result = future.result()
                if result is not None and result['passes_filter']:
                    # Add mean reversion metrics to original basket result
                    basket_idx = None
                    for i, b in enumerate(baskets):
                        if b['basket'] == result['basket']:
                            basket_idx = i
                            break
                    
                    if basket_idx is not None:
                        baskets[basket_idx]['mean_reversion'] = {
                            'half_life_periods': result['half_life_periods'],
                            'half_life_days': result['half_life_days'],
                            'adf_statistic': result['adf_statistic'],
                            'adf_p_value': result['adf_p_value'],
                            'is_stationary': result['is_stationary'],
                            'half_life_threshold_days': half_life_threshold_days
                        }
                        results.append(baskets[basket_idx])
                
            except Exception as e:
                basket = future_to_basket[future]
                logger.warning(f"Failed to test basket {basket}: {e}")
            
            if completed % 10 == 0:
                logger.info(f"Processed {completed}/{len(baskets)} baskets, "
                           f"found {len(results)} fast mean-reverting so far")
    
    logger.info(f"Filtered to {len(results)} fast mean-reverting baskets "
               f"(half-life < {half_life_threshold_days} days OR ADF p < 0.01)")
    
    return results

