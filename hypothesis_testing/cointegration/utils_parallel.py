"""
Parallel processing utilities for cointegration testing.
"""

from typing import List, Dict, Optional, Set
import numpy as np
import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from .johansen_test import johansen_test
from .basket_generator import compute_spread
from .deduplicate_baskets import filter_overlapping_baskets
from .hmm_regimes import apply_regime_filter

logger = logging.getLogger(__name__)


def _test_baskets_batch(args):
    """
    Worker function for testing multiple baskets in batch.
    Reconstructs DataFrame once per worker, then tests all baskets.
    Designed to be picklable for multiprocessing.
    """
    price_data_dict, baskets_batch, labels_dict, include_states, policy, k, coverage_threshold = args
    
    try:
        # Reconstruct price data DataFrame once per worker (not per basket!)
        price_data = pd.DataFrame(price_data_dict['data'], 
                                  index=pd.DatetimeIndex(price_data_dict['index']),
                                  columns=price_data_dict['columns'])
        
        # Reconstruct labels DataFrame once per worker if provided
        labels_df = None
        if labels_dict is not None:
            index_values = labels_dict['index']
            tz = labels_dict.get('tz')
            index = pd.DatetimeIndex(index_values)
            if tz is not None and index.tz is None:
                index = index.tz_localize(tz)
            labels_df = pd.DataFrame(labels_dict['data'], index=index, columns=labels_dict['columns'])

        results = []
        for basket in baskets_batch:
            try:
                # Extract basket prices
                basket_cols = [f'{sym}_close' for sym in basket]
                price_df = price_data
                if labels_df is not None and include_states is not None:
                    price_df, coverage = apply_regime_filter(
                        price_data=price_data,
                        labels_df=labels_df,
                        basket=basket,
                        include_states=include_states,
                        policy=policy,
                        k=k,
                        coverage_threshold=coverage_threshold,
                    )
                    if price_df.empty:
                        continue

                basket_prices = price_df[basket_cols].values
                # Check for valid data
                if np.any(basket_prices <= 0) or np.any(np.isnan(basket_prices)) or np.any(np.isinf(basket_prices)):
                    continue
                if basket_prices.shape[0] < len(basket) * 10:
                    continue

                # Convert to log prices
                log_prices = np.log(basket_prices)
                
                # Validate log prices after conversion (regime filtering may have left sparse data)
                if np.any(np.isnan(log_prices)) or np.any(np.isinf(log_prices)):
                    continue
                if log_prices.shape[0] < len(basket) * 10:
                    continue

                # Run Johansen test
                result = johansen_test(log_prices, p_value_threshold=0.01)

                if not result['is_cointegrated']:
                    continue

                # Compute spread using first eigenvector
                eigenvector = np.array(result['eigenvectors'])[:, 0]
                # Normalize by sum of absolute values (consistent with strategy/position manager)
                # This ensures eigenvector weights sum to 1 in absolute terms
                sum_abs = np.sum(np.abs(eigenvector))
                if sum_abs > 0:
                    eigenvector = eigenvector / sum_abs
                else:
                    continue  # Skip if eigenvector is invalid

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
                                        *,
                                        regime_labels: Optional[pd.DataFrame] = None,
                                        include_states: Optional[Set[int]] = None,
                                        policy: str = 'all',
                                        k: Optional[int] = None,
                                        coverage_threshold: Optional[float] = 0.4,
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

    labels_dict = None
    if regime_labels is not None:
        index_values = regime_labels.index.view('int64')
        labels_dict = {
            'data': regime_labels.values,
            'index': index_values,
            'tz': getattr(regime_labels.index, 'tz', None),
            'columns': regime_labels.columns.tolist()
        }
    
    # Batch baskets to reduce DataFrame reconstruction overhead
    batches = []
    for i in range(0, len(candidate_baskets), batch_size):
        batches.append(candidate_baskets[i:i + batch_size])
    
    # Prepare arguments (price_data_dict, baskets_batch, labels_dict, include_states, policy, k)
    args_list = [
        (price_data_dict, batch, labels_dict, include_states, policy, k, coverage_threshold)
        for batch in batches
    ]
    
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

