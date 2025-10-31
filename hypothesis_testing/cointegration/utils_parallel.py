"""
Parallel processing utilities for cointegration testing.
"""

from typing import List, Dict, Optional
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from .persistence import test_persistence_rolling
from .spread_validator import compute_max_zscore

logger = logging.getLogger(__name__)


def _test_basket_constraints(args):
    """
    Worker function for testing basket constraints.
    Designed to be picklable for multiprocessing.
    """
    basket_result, basket_size = args
    
    try:
        # Convert numpy arrays to ensure proper pickling
        import numpy as np
        log_prices = np.array(basket_result['log_prices'])
        eigenvector = np.array(basket_result['eigenvector'])
        spread = np.array(basket_result['spread'])
        
        # Check max z-score first (cheaper)
        max_z = compute_max_zscore(spread, lookback_days=90)
        
        # Size-dependent thresholds
        persistence_threshold = 0.8 - 0.1 * max(0, basket_size - 2)
        z_threshold = 3.0 + 0.5 * max(0, basket_size - 2)
        
        # Early termination if z-score fails
        if max_z >= z_threshold:
            return None
        
        # Test persistence (more expensive)
        persistence = test_persistence_rolling(log_prices, eigenvector, window_days=90, step_days=30)
        
        # Final check
        if persistence > persistence_threshold:
            # Create new dict with results (don't modify original)
            result = {
                'basket': basket_result['basket'],
                'johansen_result': basket_result['johansen_result'],
                'eigenvector': eigenvector,
                'spread': spread,
                'log_prices': log_prices,
                'persistence': persistence,
                'max_zscore': max_z,
                'persistence_threshold': persistence_threshold,
                'z_threshold': z_threshold
            }
            return result
        
        return None
        
    except Exception as e:
        logger.debug(f"Error testing basket {basket_result.get('basket', 'unknown')}: {e}")
        return None


def test_baskets_parallel(cointegrated_baskets: List[Dict], max_workers: Optional[int] = None) -> List[Dict]:
    """
    Test persistence and spread constraints in parallel.
    
    Parameters:
    -----------
    cointegrated_baskets : List[Dict]
        List of basket results from initial cointegration test
    max_workers : Optional[int]
        Maximum number of worker processes. If None, uses CPU count.
        
    Returns:
    --------
    List[Dict]
        List of baskets that passed all constraints
    """
    if not cointegrated_baskets:
        return []
    
    # Prepare arguments (basket_result, basket_size)
    args_list = [(basket_result, len(basket_result['basket'])) 
                 for basket_result in cointegrated_baskets]
    
    valid_baskets = []
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_basket = {
            executor.submit(_test_basket_constraints, args): args[0]['basket']
            for args in args_list
        }
        
        # Collect results as they complete
        completed = 0
        total = len(args_list)
        
        for future in as_completed(future_to_basket):
            completed += 1
            try:
                result = future.result()
                if result is not None:
                    valid_baskets.append(result)
            except Exception as e:
                basket = future_to_basket[future]
                logger.warning(f"Failed to test basket {basket}: {e}")
            
            if completed % 10 == 0:
                logger.info(f"Processed {completed}/{total} baskets, found {len(valid_baskets)} valid so far")
    
    return valid_baskets

