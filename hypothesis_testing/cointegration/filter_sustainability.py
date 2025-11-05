"""
Filter baskets by cointegration sustainability across time periods.
Tests both rolling windows and discrete periods with multiprocessing.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import os

from .johansen_test import johansen_test

logger = logging.getLogger(__name__)


def _test_window_cointegration(args):
    """
    Worker function to test cointegration in a single window.
    Designed to be picklable for multiprocessing.
    """
    log_prices_array, start_idx, end_idx, window_id = args
    
    try:
        window_data = log_prices_array[start_idx:end_idx]
        
        if window_data.shape[0] < window_data.shape[1] * 10:  # Need enough data
            return {'window_id': window_id, 'is_cointegrated': False, 'p_value': 1.0}
        
        result = johansen_test(window_data)
        return {
            'window_id': window_id,
            'is_cointegrated': result['is_cointegrated'],
            'p_value': result['p_value'],
            'trace_stat': result['trace_stat']
        }
    except Exception as e:
        logger.debug(f"Error testing window {window_id}: {e}")
        return {'window_id': window_id, 'is_cointegrated': False, 'p_value': 1.0}


def _test_period_cointegration(args):
    """
    Worker function to test cointegration in a discrete period.
    Designed to be picklable for multiprocessing.
    """
    log_prices_array, start_idx, end_idx, period_id = args
    
    try:
        period_data = log_prices_array[start_idx:end_idx]
        
        if period_data.shape[0] < period_data.shape[1] * 10:  # Need enough data
            return {'period_id': period_id, 'is_cointegrated': False, 'p_value': 1.0}
        
        result = johansen_test(period_data)
        return {
            'period_id': period_id,
            'is_cointegrated': result['is_cointegrated'],
            'p_value': result['p_value'],
            'trace_stat': result['trace_stat']
        }
    except Exception as e:
        logger.debug(f"Error testing period {period_id}: {e}")
        return {'period_id': period_id, 'is_cointegrated': False, 'p_value': 1.0}


def test_rolling_windows(log_prices: np.ndarray, 
                        window_days: int = 90,
                        step_days: int = 30,
                        bars_per_day: int = 24,
                        max_workers: Optional[int] = None) -> Dict:
    """
    Test cointegration persistence over rolling windows.
    
    Parameters:
    -----------
    log_prices : np.ndarray, shape (T, n)
        Log prices for n assets over T time periods
    window_days : int
        Size of rolling window in days
    step_days : int
        Step size between windows in days
    bars_per_day : int
        Number of bars per day (24 for 1h, 96 for 15m)
    max_workers : Optional[int]
        Number of parallel workers. If None, uses CPU count.
        
    Returns:
    --------
    dict with keys: 'persistence_ratio', 'windows_passed', 'total_windows', 'window_results'
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    window_bars = window_days * bars_per_day
    step_bars = step_days * bars_per_day
    
    T, n = log_prices.shape
    if T < window_bars:
        return {
            'persistence_ratio': 0.0,
            'windows_passed': 0,
            'total_windows': 0,
            'window_results': []
        }
    
    # Prepare window arguments for parallel processing
    windows = []
    window_id = 0
    for start in range(0, T - window_bars + 1, step_bars):
        end = start + window_bars
        windows.append((log_prices, start, end, window_id))
        window_id += 1
    
    if not windows:
        return {
            'persistence_ratio': 0.0,
            'windows_passed': 0,
            'total_windows': 0,
            'window_results': []
        }
    
    # Test windows in parallel
    window_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_window = {
            executor.submit(_test_window_cointegration, args): args[3]
            for args in windows
        }
        
        completed = 0
        for future in as_completed(future_to_window):
            completed += 1
            try:
                result = future.result()
                window_results.append(result)
            except Exception as e:
                window_id = future_to_window[future]
                logger.warning(f"Failed to test window {window_id}: {e}")
    
    # Sort by window_id to maintain order
    window_results.sort(key=lambda x: x['window_id'])
    
    # Calculate persistence ratio
    windows_passed = sum(r['is_cointegrated'] for r in window_results)
    total_windows = len(window_results)
    persistence_ratio = windows_passed / total_windows if total_windows > 0 else 0.0
    
    return {
        'persistence_ratio': persistence_ratio,
        'windows_passed': windows_passed,
        'total_windows': total_windows,
        'window_results': window_results
    }


def test_discrete_periods(log_prices: np.ndarray,
                         period_days: int = 90,
                         bars_per_day: int = 24,
                         max_workers: Optional[int] = None) -> Dict:
    """
    Test cointegration across discrete non-overlapping periods.
    
    Parameters:
    -----------
    log_prices : np.ndarray, shape (T, n)
        Log prices for n assets over T time periods
    period_days : int
        Size of each discrete period in days
    bars_per_day : int
        Number of bars per day (24 for 1h, 96 for 15m)
    max_workers : Optional[int]
        Number of parallel workers. If None, uses CPU count.
        
    Returns:
    --------
    dict with keys: 'persistence_ratio', 'periods_passed', 'total_periods', 'period_results'
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    period_bars = period_days * bars_per_day
    
    T, n = log_prices.shape
    if T < period_bars:
        return {
            'persistence_ratio': 0.0,
            'periods_passed': 0,
            'total_periods': 0,
            'period_results': []
        }
    
    # Prepare period arguments for parallel processing
    periods = []
    period_id = 0
    for start in range(0, T - period_bars + 1, period_bars):
        end = start + period_bars
        periods.append((log_prices, start, end, period_id))
        period_id += 1
    
    if not periods:
        return {
            'persistence_ratio': 0.0,
            'periods_passed': 0,
            'total_periods': 0,
            'period_results': []
        }
    
    # Test periods in parallel
    period_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_period = {
            executor.submit(_test_period_cointegration, args): args[3]
            for args in periods
        }
        
        completed = 0
        for future in as_completed(future_to_period):
            completed += 1
            try:
                result = future.result()
                period_results.append(result)
            except Exception as e:
                period_id = future_to_period[future]
                logger.warning(f"Failed to test period {period_id}: {e}")
    
    # Sort by period_id to maintain order
    period_results.sort(key=lambda x: x['period_id'])
    
    # Calculate persistence ratio
    periods_passed = sum(r['is_cointegrated'] for r in period_results)
    total_periods = len(period_results)
    persistence_ratio = periods_passed / total_periods if total_periods > 0 else 0.0
    
    return {
        'persistence_ratio': persistence_ratio,
        'periods_passed': periods_passed,
        'total_periods': total_periods,
        'period_results': period_results
    }


def filter_baskets_sustainability(baskets: List[Dict],
                                 price_data: pd.DataFrame,
                                 persistence_threshold: float = 0.7,
                                 window_days: int = 90,
                                 step_days: int = 30,
                                 period_days: int = 90,
                                 bars_per_day: int = 24,
                                 use_rolling: bool = True,
                                 use_discrete: bool = True,
                                 max_workers: Optional[int] = None) -> List[Dict]:
    """
    Filter baskets by cointegration sustainability across time periods.
    
    Parameters:
    -----------
    baskets : List[Dict]
        List of basket dictionaries with keys: 'basket', 'johansen_result', 'eigenvector', 'spread', 'log_prices'
    price_data : pd.DataFrame
        Price data with columns {symbol}_close
    persistence_threshold : float
        Minimum persistence ratio to pass (default 0.7 = 70%)
    window_days : int
        Size of rolling window in days
    step_days : int
        Step size between rolling windows in days
    period_days : int
        Size of discrete periods in days
    bars_per_day : int
        Number of bars per day (24 for 1h, 96 for 15m)
    use_rolling : bool
        Whether to test rolling windows
    use_discrete : bool
        Whether to test discrete periods
    max_workers : Optional[int]
        Number of parallel workers. If None, uses CPU count.
        
    Returns:
    --------
    List[Dict]
        Filtered baskets with added 'sustainability' metrics
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    logger.info(f"Testing {len(baskets)} baskets for sustainability...")
    
    sustainable_baskets = []
    
    for i, basket_result in enumerate(baskets):
        basket = basket_result['basket']
        log_prices = basket_result['log_prices']
        
        # Test rolling windows
        rolling_result = None
        if use_rolling:
            rolling_result = test_rolling_windows(
                log_prices, window_days, step_days, bars_per_day, max_workers
            )
        
        # Test discrete periods
        discrete_result = None
        if use_discrete:
            discrete_result = test_discrete_periods(
                log_prices, period_days, bars_per_day, max_workers
            )
        
        # Determine if basket passes threshold
        passes = False
        if use_rolling and use_discrete:
            # Both must pass threshold
            passes = (rolling_result['persistence_ratio'] >= persistence_threshold and
                     discrete_result['persistence_ratio'] >= persistence_threshold)
        elif use_rolling:
            passes = rolling_result['persistence_ratio'] >= persistence_threshold
        elif use_discrete:
            passes = discrete_result['persistence_ratio'] >= persistence_threshold
        
        if passes:
            basket_result['sustainability'] = {
                'rolling_windows': rolling_result,
                'discrete_periods': discrete_result,
                'persistence_threshold': persistence_threshold
            }
            sustainable_baskets.append(basket_result)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(baskets)} baskets, "
                       f"found {len(sustainable_baskets)} sustainable so far")
    
    logger.info(f"Filtered to {len(sustainable_baskets)} sustainable baskets "
               f"(threshold: {persistence_threshold:.0%})")
    
    return sustainable_baskets

