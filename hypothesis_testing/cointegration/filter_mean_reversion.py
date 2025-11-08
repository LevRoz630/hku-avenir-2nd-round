"""
Filter baskets by mean reversion speed.
Computes Hurst-based half-life metrics with multiprocessing.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import os

logger = logging.getLogger(__name__)


def _compute_half_life(spread: np.ndarray) -> float:
    """
    Compute half-life using Hurst exponent for crypto spread mean reversion.

    Hurst exponent (H) measures long-term memory and persistence:
    - H < 0.5: mean reverting (good for pairs trading)
    - H = 0.5: random walk (neutral)
    - H > 0.5: trending/persistent (bad for pairs trading)

    For crypto spreads, Hurst is more robust than OU process because:
    - No assumptions about constant mean reversion speed
    - Handles non-stationary volatility and structural breaks
    - Measures true long-term persistence, not just short-term autocorrelation

    Half-life derived from Hurst: stronger mean reversion (lower H) = shorter half-life

    Parameters:
    -----------
    spread : np.ndarray, shape (T,)
        Spread series

    Returns:
    --------
    float
        Half-life in periods (or np.inf if not mean reverting)
    """
    if len(spread) < 50:  # Need sufficient data for Hurst estimation
        return np.inf

    try:
        return _compute_half_life_hurst(spread)
    except:
        logger.debug("Error in half-life calculation")
        return np.inf


def _compute_half_life_hurst(spread: np.ndarray) -> float:
    """Hurst exponent based half-life estimation."""
    if len(spread) < 50:
        return np.inf

    try:
        # Simplified Hurst exponent calculation
        # H = 0.5: random walk (no mean reversion)
        # H < 0.5: mean reverting
        # H > 0.5: trending

        spread_centered = spread - np.mean(spread)
        cumulative = np.cumsum(spread_centered)

        # R/S analysis for different window sizes
        # Use more window sizes for better estimation, but ensure minimum size
        min_window = max(10, len(spread) // 20)  # At least 5% of data
        max_window = len(spread) // 2  # At most 50% of data
        if max_window < min_window:
            return np.inf
        
        # Generate window sizes logarithmically spaced
        num_windows = min(8, max(3, int(np.log2(max_window / min_window)) + 1))
        window_sizes = np.logspace(np.log2(min_window), np.log2(max_window), num_windows, base=2).astype(int)
        window_sizes = np.unique(window_sizes)  # Remove duplicates

        rs_values = []
        for window in window_sizes:
            rs_window = []
            # Use non-overlapping windows for R/S analysis
            for i in range(0, len(cumulative) - window + 1, window):
                segment = cumulative[i:i+window]
                if len(segment) >= min(10, window // 2):
                    R = np.max(segment) - np.min(segment)
                    S = np.std(segment, ddof=1)
                    if S > 1e-10:  # Avoid division by very small numbers
                        rs_window.append(R / S)
            if len(rs_window) >= 2:  # Need at least 2 windows for reliable estimate
                rs_values.append(np.mean(rs_window))

        if len(rs_values) >= 2:
            # Fit log-log regression to estimate Hurst exponent
            x = np.log(window_sizes[:len(rs_values)])
            y = np.log(rs_values)
            if len(x) >= 2 and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                slope = np.polyfit(x, y, 1)[0]
                hurst = slope

                # For mean reverting series (H < 0.5), estimate half-life
                # Using autocorrelation-based approximation: half-life ≈ -ln(2) / ln(ρ)
                # For Hurst H < 0.5, we approximate: stronger mean reversion (lower H) = shorter half-life
                if hurst < 0.5:
                    # More conservative formula: half-life increases exponentially as H approaches 0.5
                    # For H=0.3: half-life ≈ 20 periods, for H=0.4: half-life ≈ 50 periods
                    half_life = max(1, int(100 * (0.5 - hurst)))
                    return half_life
    except:
        logger.debug("Error in half-life calculation")

    return np.inf




def _test_basket_mean_reversion(args):
    """
    Worker function to compute mean reversion metrics for a single basket.
    Uses test_data to recompute spread (prevents data leakage).
    Designed to be picklable for multiprocessing.
    """
    basket_result, test_data_dict, half_life_threshold_days, bars_per_day = args
    
    try:
        # Reconstruct test_data DataFrame
        index_values = test_data_dict['index']
        tz = test_data_dict.get('tz')
        index = pd.DatetimeIndex(index_values)
        if tz is not None and index.tz is None:
            index = index.tz_localize(tz)
        test_data = pd.DataFrame(
            test_data_dict['data'],
            index=index,
            columns=test_data_dict['columns']
        )
        
        # Extract basket prices from test_data
        basket = basket_result['basket']
        basket_cols = [f'{sym}_close' for sym in basket]
        
        if not all(col in test_data.columns for col in basket_cols):
            return None
        
        basket_prices = test_data[basket_cols].values
        
        # Validate prices
        if np.any(basket_prices <= 0):
            logger.debug(f"Invalid prices (non-positive) for basket {basket}")
            return None
        
        if np.any(np.isnan(basket_prices)) or np.any(np.isinf(basket_prices)):
            logger.debug(f"Invalid prices (NaN/Inf) for basket {basket}")
            return None
        
        # Convert to log prices
        log_prices = np.log(basket_prices)
        
        # Validate log prices
        if np.any(np.isnan(log_prices)) or np.any(np.isinf(log_prices)):
            logger.debug(f"Invalid log prices (NaN/Inf) for basket {basket}")
            return None
        
        # Recompute spread using eigenvector from cointegration test
        eigenvector = np.array(basket_result['eigenvector'])
        
        if np.any(np.isnan(eigenvector)) or np.any(np.isinf(eigenvector)):
            logger.debug(f"Invalid eigenvector (NaN/Inf) for basket {basket}")
            return None
        
        spread = log_prices @ eigenvector
        
        # Validate spread
        if np.any(np.isnan(spread)) or np.any(np.isinf(spread)):
            logger.debug(f"Invalid spread (NaN/Inf) for basket {basket}")
            return None
        
        # Compute half-life using Hurst exponent
        half_life_periods = _compute_half_life(spread)
        half_life_days = half_life_periods / bars_per_day

        # Determine if basket passes filter (Hurst-based half-life only)
        passes = half_life_days < half_life_threshold_days

        return {
            'basket': basket_result['basket'],
            'half_life_periods': half_life_periods,
            'half_life_days': half_life_days,
            'passes_filter': passes
        }
    except Exception as e:
        logger.debug(f"Error testing basket {basket_result.get('basket', 'unknown')}: {e}")
        return None


def filter_baskets_mean_reversion(baskets: List[Dict],
                                  test_data: pd.DataFrame,
                                  half_life_threshold_days: float = 30.0,
                                  bars_per_day: int = 24,
                                  max_workers: Optional[int] = None) -> List[Dict]:
    """
    Filter baskets by mean reversion speed.
    Uses separate test_data to recompute spreads (prevents data leakage).
    
    Parameters:
    -----------
    baskets : List[Dict]
        List of basket dictionaries with 'eigenvector' key (from cointegration test)
    test_data : pd.DataFrame
        Price data for testing (should be separate from cointegration data)
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

    if bars_per_day <= 0:
        raise ValueError("bars_per_day must be positive")

    logger.info(f"Testing {len(baskets)} baskets for mean reversion speed "
               f"(using separate test data: {len(test_data)} samples, "
               f"parallelized with {max_workers} workers)...")
    
    # Convert DataFrame to dict for pickling
    index_values = test_data.index.view('int64')
    test_data_dict = {
        'data': test_data.values,
        'index': index_values,
        'tz': getattr(test_data.index, 'tz', None),
        'columns': test_data.columns.tolist()
    }
    
    # Create basket index for O(1) lookup
    basket_to_idx = {tuple(b['basket']): i for i, b in enumerate(baskets)}
    
    # Prepare arguments for parallel processing
    args_list = [
        (basket_result, test_data_dict, half_life_threshold_days, bars_per_day)
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
                    # Add mean reversion metrics to original basket result (O(1) lookup)
                    basket_key = tuple(result['basket'])
                    basket_idx = basket_to_idx.get(basket_key)
                    
                    if basket_idx is not None:
                        baskets[basket_idx]['mean_reversion'] = {
                            'half_life_periods': result['half_life_periods'],
                            'half_life_days': result['half_life_days'],
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
               f"(Hurst-based half-life < {half_life_threshold_days} days)")
    
    return results

