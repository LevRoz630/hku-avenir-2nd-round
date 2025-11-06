"""
Parallel processing utilities for cointegration testing.
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from .johansen_test import johansen_test
from .basket_generator import compute_spread
from .deduplicate_baskets import filter_overlapping_baskets

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
                                        batch_size: int = 100,
                                        deduplicate: bool = True,
                                        overlap_threshold: float = 0.5,
                                        prefer_lower_pvalue: bool = True) -> List[Dict]:
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
    deduplicate : bool
        If True, remove highly overlapping baskets after testing (default: True).
    overlap_threshold : float
        Overlap ratio threshold for deduplication (default: 0.5 = 50%).
    prefer_lower_pvalue : bool
        If True, prefer baskets with lower Johansen p-value when overlaps are removed.
        
    Returns:
    --------
    List[Dict]
        List of cointegrated baskets
    """
    if not candidate_baskets:
        return []
    
    import os
    if max_workers is None:
        max_workers = int(0.9 * (os.cpu_count() or 1)) or 1
    
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
    
    if deduplicate and cointegrated_baskets:
        before = len(cointegrated_baskets)
        cointegrated_baskets = filter_overlapping_baskets(
            cointegrated_baskets,
            overlap_threshold=overlap_threshold,
            prefer_lower_pvalue=prefer_lower_pvalue,
        )
        removed = before - len(cointegrated_baskets)
        logger.info(
            "Deduplicated cointegrated baskets: %d kept, %d removed (threshold %.0f%%)",
            len(cointegrated_baskets),
            removed,
            overlap_threshold * 100,
        )

    return cointegrated_baskets

