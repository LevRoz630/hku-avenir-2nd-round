"""
Data splitting utilities for preventing overfitting in cointegration testing.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def split_data_chronologically(price_data: pd.DataFrame,
                               cluster_split: Tuple[float, float] = (1.2, 0.8),
                               cointegration_split: Tuple[float, float] = (0.8, 0.2),
                               half_life_split: Tuple[float, float] = (0.2, 0.0)) -> dict:
    """
    Split price data chronologically to prevent overfitting.
    
    Data is split by time quantiles (most recent = 1.0, oldest = 0.0):
    - Cluster analysis: Uses most recent data (default: 1.2-0.8 = top 20%, clamped to 1.0-0.8)
    - Cointegration testing: Uses middle data (default: 0.8-0.2 = middle 60%)
    - Half-life testing: Uses oldest data (default: 0.2-0.0 = bottom 20%)
    
    This prevents:
    - Overfitting: Clustering on same data as cointegration testing
    - Look-ahead bias: Using future data to select baskets
    - Data leakage: Testing on data used for training
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Price data with datetime index
    cluster_split : Tuple[float, float]
        Quantile range for cluster analysis (start, end)
        Default: (1.2, 0.8) = most recent 20% (clamped to 1.0-0.8)
    cointegration_split : Tuple[float, float]
        Quantile range for cointegration testing (start, end)
        Default: (0.8, 0.2) = middle 60%
    half_life_split : Tuple[float, float]
        Quantile range for half-life testing (start, end)
        Default: (0.2, 0.0) = oldest 20%
        
    Returns:
    --------
    dict with keys:
        - 'cluster_data': DataFrame for cluster analysis
        - 'cointegration_data': DataFrame for cointegration testing
        - 'half_life_data': DataFrame for half-life testing
        - 'splits_info': Dict with split details
    """
    if not isinstance(price_data.index, pd.DatetimeIndex):
        raise ValueError("price_data must have DatetimeIndex")
    
    # Sort by time (most recent first)
    price_data_sorted = price_data.sort_index(ascending=False)
    n_total = len(price_data_sorted)
    
    # Clamp cluster_split[0] to max 1.0 (can't go beyond most recent)
    cluster_start_quantile = min(cluster_split[0], 1.0)
    
    # Calculate split indices
    cluster_start_idx = int(n_total * (1.0 - cluster_start_quantile))
    cluster_end_idx = int(n_total * (1.0 - cluster_split[1]))
    
    cointegration_start_idx = int(n_total * (1.0 - cointegration_split[0]))
    cointegration_end_idx = int(n_total * (1.0 - cointegration_split[1]))
    
    half_life_start_idx = int(n_total * (1.0 - half_life_split[0]))
    half_life_end_idx = int(n_total * (1.0 - half_life_split[1]))
    
    # Extract splits (most recent first, so we slice from start)
    cluster_data = price_data_sorted.iloc[cluster_start_idx:cluster_end_idx].sort_index(ascending=True)
    cointegration_data = price_data_sorted.iloc[cointegration_start_idx:cointegration_end_idx].sort_index(ascending=True)
    half_life_data = price_data_sorted.iloc[half_life_start_idx:half_life_end_idx].sort_index(ascending=True)
    
    splits_info = {
        'total_samples': n_total,
        'cluster_samples': len(cluster_data),
        'cointegration_samples': len(cointegration_data),
        'half_life_samples': len(half_life_data),
        'cluster_date_range': (cluster_data.index.min(), cluster_data.index.max()),
        'cointegration_date_range': (cointegration_data.index.min(), cointegration_data.index.max()),
        'half_life_date_range': (half_life_data.index.min(), half_life_data.index.max()),
    }
    
    logger.info(f"Split data chronologically:")
    logger.info(f"  Cluster: {len(cluster_data)} samples ({cluster_split[0]:.1f}-{cluster_split[1]:.0%}) "
               f"from {cluster_data.index.min()} to {cluster_data.index.max()}")
    logger.info(f"  Cointegration: {len(cointegration_data)} samples ({cointegration_split[0]:.0%}-{cointegration_split[1]:.0%}) "
               f"from {cointegration_data.index.min()} to {cointegration_data.index.max()}")
    logger.info(f"  Half-life: {len(half_life_data)} samples ({half_life_split[0]:.0%}-{half_life_split[1]:.0%}) "
               f"from {half_life_data.index.min()} to {half_life_data.index.max()}")
    
    return {
        'cluster_data': cluster_data,
        'cointegration_data': cointegration_data,
        'half_life_data': half_life_data,
        'splits_info': splits_info
    }

