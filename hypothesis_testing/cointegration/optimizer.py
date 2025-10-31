"""
Lookback window and basket size optimization.
"""

from typing import List, Dict, Optional
import numpy as np
import logging

from .persistence import test_persistence_rolling
from .spread_validator import compute_max_zscore

logger = logging.getLogger(__name__)


def optimize_lookback_window(log_prices: np.ndarray, eigenvector: np.ndarray,
                             spread: np.ndarray, 
                             candidate_windows: List[int] = [30, 60, 90, 120, 150, 180],
                             lambda_penalty: float = 0.2,
                             basket_size: int = 2) -> Dict:
    """
    Find optimal lookback window that maximizes persistence score while keeping max z-score < threshold.
    
    Parameters:
    -----------
    log_prices : np.ndarray, shape (T, n)
        Log prices for n assets over T time periods
    eigenvector : np.ndarray, shape (n,)
        Cointegration eigenvector
    spread : np.ndarray, shape (T,)
        Spread series
    candidate_windows : List[int]
        List of candidate window sizes in days
    lambda_penalty : float
        Penalty weight for high z-scores in objective function
    basket_size : int
        Number of assets in the basket (for size-dependent thresholds)
        
    Returns:
    --------
    Dict
        Dictionary with 'optimal_window', 'optimal_metrics', and 'all_results'
    """
    # Size-dependent z-score threshold
    z_threshold = 3.0 + 0.5 * max(0, basket_size - 2)  # 3.0 for size 2, 3.5 for size 3, etc.
    
    best_window = None
    best_score = -np.inf
    best_metrics = None
    
    results = []
    
    for w in candidate_windows:
        # Compute persistence with this window
        persistence = test_persistence_rolling(log_prices, eigenvector, 
                                              window_days=w, step_days=max(30, w//3))
        
        # Compute max z-score with this lookback
        max_z = compute_max_zscore(spread, lookback_days=w)
        
        # Objective: maximize persistence, penalize high z-scores
        # Only consider windows where max_z < z_threshold (size-dependent)
        if max_z < z_threshold:
            # Score: persistence - lambda * (max_z / z_threshold)
            score = persistence - lambda_penalty * (max_z / z_threshold)
            
            if score > best_score:
                best_score = score
                best_window = w
                best_metrics = {
                    'window_days': w,
                    'persistence': persistence,
                    'max_zscore': max_z,
                    'score': score
                }
        
        results.append({
            'window_days': w,
            'persistence': persistence,
            'max_zscore': max_z,
            'valid': max_z < z_threshold
        })
    
    return {
        'optimal_window': best_window,
        'optimal_metrics': best_metrics,
        'all_results': results
    }


def score_basket(basket_result: Dict) -> float:
    """
    Score basket: persistence × (1 - max_zscore/z_threshold)
    Higher is better. Uses size-dependent z_threshold.
    
    Parameters:
    -----------
    basket_result : Dict
        Dictionary containing 'persistence', 'max_zscore', and 'basket' keys
        
    Returns:
    --------
    float
        Basket score (0.0 to 1.0)
    """
    persistence = basket_result.get('persistence', 0.0)
    max_z = basket_result.get('max_zscore', 3.0)
    basket_size = len(basket_result.get('basket', []))
    
    # Size-dependent z-score threshold
    z_threshold = 3.0 + 0.5 * max(0, basket_size - 2)
    
    # Normalize max_z to [0, 1] where 0 is best (max_z=0) and 1 is worst (max_z=z_threshold)
    z_penalty = min(max_z / z_threshold, 1.0)
    
    score = persistence * (1 - z_penalty)
    return score


def optimize_basket_size(optimized_baskets: List[Dict]) -> Dict:
    """
    Find best basket for each size and overall best basket.
    
    Parameters:
    -----------
    optimized_baskets : List[Dict]
        List of basket results with scores
        
    Returns:
    --------
    Dict
        Dictionary with 'best_by_size' and 'best_overall'
    """
    # Score all baskets
    for basket_result in optimized_baskets:
        basket_result['final_score'] = score_basket(basket_result)
    
    # Sort by score
    optimized_baskets.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Group by basket size and find best for each size
    baskets_by_size = {}
    for basket_result in optimized_baskets:
        size = len(basket_result['basket'])
        if size not in baskets_by_size or basket_result['final_score'] > baskets_by_size[size]['final_score']:
            baskets_by_size[size] = basket_result
    
    best_overall = optimized_baskets[0] if optimized_baskets else None
    
    return {
        'best_by_size': baskets_by_size,
        'best_overall': best_overall
    }

