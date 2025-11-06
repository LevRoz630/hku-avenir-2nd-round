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
                               test_split: Tuple[float, float] = (0.2, 0.0)) -> dict:
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
    test_split : Tuple[float, float]
        Quantile range for test data (start, end)
        Default: (0.2, 0.0) = oldest 20%
        
    Returns:
    --------
    dict with keys:
        - 'cluster_data': DataFrame for cluster analysis
        - 'cointegration_data': DataFrame for cointegration testing
        - 'test_data': DataFrame for testing
        - 'splits_info': Dict with split details
    """
    if not isinstance(price_data.index, pd.DatetimeIndex):
        raise ValueError("price_data must have DatetimeIndex")
    
    # Sort by time (most recent first)
    price_data_sorted = price_data.sort_index(ascending=False)
    n_total = len(price_data_sorted)
    
    if n_total == 0:
        raise ValueError("price_data is empty")
    
    # Validate split ranges don't overlap incorrectly
    if cluster_split[1] > cointegration_split[0]:
        raise ValueError(f"Cluster and cointegration splits overlap: cluster ends at {cluster_split[1]}, cointegration starts at {cointegration_split[0]}")
    
    if cointegration_split[1] > test_split[0]:
        raise ValueError(f"Cointegration and test splits overlap: cointegration ends at {cointegration_split[1]}, test starts at {test_split[0]}")
    
    # Clamp cluster_split[0] to max 1.0 (can't go beyond most recent)
    cluster_start_quantile = min(cluster_split[0], 1.0)
    
    # Calculate split indices
    cluster_start_idx = int(n_total * (1.0 - cluster_start_quantile))
    cluster_end_idx = int(n_total * (1.0 - cluster_split[1]))
    
    cointegration_start_idx = int(n_total * (1.0 - cointegration_split[0]))
    cointegration_end_idx = int(n_total * (1.0 - cointegration_split[1]))
    
    test_start_idx = int(n_total * (1.0 - test_split[0]))
    test_end_idx = int(n_total * (1.0 - test_split[1]))
    
    # Validate indices are valid
    if cluster_start_idx < 0 or cluster_end_idx > n_total or cluster_start_idx >= cluster_end_idx:
        raise ValueError(f"Invalid cluster split indices: start={cluster_start_idx}, end={cluster_end_idx}, total={n_total}")
    
    if cointegration_start_idx < 0 or cointegration_end_idx > n_total or cointegration_start_idx >= cointegration_end_idx:
        raise ValueError(f"Invalid cointegration split indices: start={cointegration_start_idx}, end={cointegration_end_idx}, total={n_total}")
    
    if test_start_idx < 0 or test_end_idx > n_total or test_start_idx >= test_end_idx:
        raise ValueError(f"Invalid test split indices: start={test_start_idx}, end={test_end_idx}, total={n_total}")
    
    # Extract splits (most recent first, so we slice from start)
    cluster_data = price_data_sorted.iloc[cluster_start_idx:cluster_end_idx].sort_index(ascending=True)
    cointegration_data = price_data_sorted.iloc[cointegration_start_idx:cointegration_end_idx].sort_index(ascending=True)
    test_data = price_data_sorted.iloc[test_start_idx:test_end_idx].sort_index(ascending=True)
    
    splits_info = {
        'total_samples': n_total,
        'cluster_samples': len(cluster_data),
        'cointegration_samples': len(cointegration_data),
        'test_samples': len(test_data),
        'cluster_date_range': (cluster_data.index.min(), cluster_data.index.max()),
        'cointegration_date_range': (cointegration_data.index.min(), cointegration_data.index.max()),
        'test_date_range': (test_data.index.min(), test_data.index.max()),
    }
    
    logger.info(f"Split data chronologically:")
    logger.info(f"  Cluster: {len(cluster_data)} samples ({cluster_split[0]:.1f}-{cluster_split[1]:.0%}) "
               f"from {cluster_data.index.min()} to {cluster_data.index.max()}")
    logger.info(f"  Cointegration: {len(cointegration_data)} samples ({cointegration_split[0]:.0%}-{cointegration_split[1]:.0%}) "
               f"from {cointegration_data.index.min()} to {cointegration_data.index.max()}")
    logger.info(f"  Test: {len(test_data)} samples ({test_split[0]:.0%}-{test_split[1]:.0%}) "
               f"from {test_data.index.min()} to {test_data.index.max()}")
    
    return {
        'cluster_data': cluster_data,
        'cointegration_data': cointegration_data,
        'test_data': test_data,
        'splits_info': splits_info
    }

