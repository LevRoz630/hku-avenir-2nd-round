"""
Deduplicate baskets by removing those with high overlap with better baskets.
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def filter_overlapping_baskets(baskets: List[Dict],
                               overlap_threshold: float = 0.5,
                               prefer_lower_pvalue: bool = True) -> List[Dict]:
    """
    Remove baskets that have high overlap with other baskets.
    Keeps baskets with better cointegration stats (lower p-value by default).
    
    Algorithm:
    1. Sort baskets by quality (p-value ascending = better first)
    2. For each basket, check if it overlaps significantly with any kept basket
    3. If overlap >= threshold, skip it; otherwise keep it
    
    Parameters:
    -----------
    baskets : List[Dict]
        List of basket dictionaries with 'basket' and 'johansen_result' keys
    overlap_threshold : float
        Minimum overlap ratio to consider baskets duplicates (default: 0.5 = 50%)
        E.g., if basket A has 2/3 coins in basket B, overlap = 0.67
    prefer_lower_pvalue : bool
        If True, keep baskets with lower p-values when duplicates found
        If False, keep baskets with higher trace statistics
        
    Returns:
    --------
    List[Dict]
        Deduplicated baskets (only unique baskets with low overlap)
    """
    if not baskets:
        return []
    
    # Sort by quality: lower p-value = better (or higher trace stat if p-values equal)
    sorted_baskets = sorted(
        baskets,
        key=lambda b: (
            b['johansen_result']['p_value'],
            -b['johansen_result']['trace_stat']  # Higher trace stat is better
        )
    )
    
    kept_baskets = []
    kept_basket_sets = []  # Cache sets for faster lookup
    
    for basket in sorted_baskets:
        basket_set = set(basket['basket'])
        basket_size = len(basket_set)
        
        # Check overlap with all kept baskets
        is_duplicate = False
        for kept_set, kept_basket in zip(kept_basket_sets, kept_baskets):
            intersection = len(basket_set & kept_set)
            min_size = min(basket_size, len(kept_set))
            
            if min_size > 0:
                overlap = intersection / min_size
                if overlap >= overlap_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            kept_baskets.append(basket)
            kept_basket_sets.append(basket_set)
    
    removed_count = len(baskets) - len(kept_baskets)
    logger.info(f"Deduplicated baskets: {len(kept_baskets)} kept, {removed_count} removed "
               f"(overlap threshold: {overlap_threshold:.0%})")
    
    return kept_baskets

