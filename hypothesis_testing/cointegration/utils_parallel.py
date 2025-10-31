"""
Parallel processing utilities for cointegration testing.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from .persistence import test_persistence_rolling
from .spread_validator import compute_max_zscore
from .johansen_test import johansen_test
from .basket_generator import compute_spread

logger = logging.getLogger(__name__)


def _test_baskets_batch(args):
    """
    Worker function for testing multiple baskets in batch.
    Reconstructs DataFrame once per worker, then tests all baskets.
    Designed to be picklable for multiprocessing.
    """
    price_data_dict, baskets_batch = args
    
    try:
        # Reconstruct price data DataFrame once per worker (not per basket!)
        price_data = pd.DataFrame(price_data_dict['data'], 
                                  index=pd.DatetimeIndex(price_data_dict['index']),
                                  columns=price_data_dict['columns'])
        
        results = []
        for basket in baskets_batch:
            try:
                # Extract basket prices
                basket_cols = [f'{sym}_close' for sym in basket]
                basket_prices = price_data[basket_cols].values
                
                # Convert to log prices
                log_prices = np.log(basket_prices)
                
                # Run Johansen test
                result = johansen_test(log_prices)
                
                if not result['is_cointegrated']:
                    continue
                
                # Compute spread using first eigenvector (normalized)
                eigenvector = result['eigenvectors'][:, 0]
                # Normalize so first element is 1 (standard normalization)
                eigenvector = eigenvector / eigenvector[0]
                
                spread = compute_spread(log_prices, eigenvector)
                
                results.append({
                    'basket': basket,
                    'johansen_result': result,
                    'eigenvector': eigenvector,
                    'spread': spread,
                    'log_prices': log_prices
                })
            except Exception as e:
                logger.debug(f"Error testing basket {basket}: {e}")
                continue
        
        return results
        
    except Exception as e:
        logger.warning(f"Error in batch worker: {e}")
        return []


def test_baskets_cointegration_parallel(price_data: pd.DataFrame, 
                                        candidate_baskets: List[List[str]], 
                                        max_workers: Optional[int] = None,
                                        batch_size: int = 100) -> List[Dict]:
    """
    Test baskets for cointegration in parallel with batching.
    Batches baskets per worker to reduce DataFrame reconstruction overhead.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Price data with columns {symbol}_close
    candidate_baskets : List[List[str]]
        List of candidate baskets to test
    max_workers : Optional[int]
        Maximum number of worker processes. If None, uses CPU count.
    batch_size : int
        Number of baskets to test per worker batch. Larger = less overhead but less parallelism.
        
    Returns:
    --------
    List[Dict]
        List of cointegrated baskets
    """
    if not candidate_baskets:
        return []
    
    import os
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    # Convert DataFrame to dict for pickling (DataFrames can have pickling issues)
    price_data_dict = {
        'data': price_data.values,
        'index': price_data.index.values,
        'columns': price_data.columns.tolist()
    }
    
    # Batch baskets to reduce DataFrame reconstruction overhead
    batches = []
    for i in range(0, len(candidate_baskets), batch_size):
        batches.append(candidate_baskets[i:i + batch_size])
    
    # Prepare arguments (price_data_dict, baskets_batch)
    args_list = [(price_data_dict, batch) for batch in batches]
    
    cointegrated_baskets = []
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(_test_baskets_batch, args): i
            for i, args in enumerate(args_list)
        }
        
        # Collect results as they complete
        completed = 0
        total = len(batches)
        
        for future in as_completed(future_to_batch):
            completed += 1
            try:
                batch_results = future.result()
                cointegrated_baskets.extend(batch_results)
            except Exception as e:
                batch_idx = future_to_batch[future]
                logger.warning(f"Failed to test batch {batch_idx}: {e}")
            
            if completed % 10 == 0:
                logger.info(f"Processed {completed}/{total} batches ({completed * batch_size} baskets), "
                           f"found {len(cointegrated_baskets)} cointegrated so far")
    
    return cointegrated_baskets


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

