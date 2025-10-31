"""
Basket generation and initial cointegration screening.
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
import logging

from .johansen_test import johansen_test

logger = logging.getLogger(__name__)


def generate_baskets_clustering(price_data: pd.DataFrame, n_clusters: int = 5, 
                                min_basket_size: int = 2, max_basket_size: int = 6) -> List[List[str]]:
    """
    Generate candidate baskets using hierarchical clustering on correlation matrix.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Price data with columns {symbol}_close
    n_clusters : int
        Number of clusters for hierarchical clustering
    min_basket_size : int
        Minimum number of assets in a basket
    max_basket_size : int
        Maximum number of assets in a basket
        
    Returns:
    --------
    List[List[str]]
        List of candidate baskets, each basket is a list of symbol names
    """
    # Compute correlation matrix of returns
    returns = price_data.pct_change().dropna()
    corr_matrix = returns.corr().values
    
    # Use clustering to find groups
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    labels = clustering.fit_predict(corr_matrix)
    
    # Group symbols by cluster
    clusters = {}
    for i, symbol in enumerate(price_data.columns):
        cluster_id = labels[i]
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(symbol.replace('_close', ''))
    
    # Generate baskets from clusters
    baskets = []
    for cluster_symbols in clusters.values():
        if len(cluster_symbols) < min_basket_size:
            continue
        # Generate all combinations within cluster
        for r in range(min_basket_size, min(len(cluster_symbols) + 1, max_basket_size + 1)):
            baskets.extend([list(combo) for combo in combinations(cluster_symbols, r)])
    
    return baskets


def compute_spread(log_prices: np.ndarray, eigenvector: np.ndarray) -> np.ndarray:
    """
    Compute spread series from log prices and cointegration eigenvector.
    spread_t = Σ w_i * log(price_i,t) where w is the eigenvector
    
    Parameters:
    -----------
    log_prices : np.ndarray, shape (T, n)
        Log prices for n assets over T time periods
    eigenvector : np.ndarray, shape (n,)
        Cointegration eigenvector (weights)
        
    Returns:
    --------
    np.ndarray, shape (T,)
        Spread series
    """
    return log_prices @ eigenvector


def test_basket_cointegration(price_data: pd.DataFrame, basket: List[str]) -> Optional[Dict]:
    """
    Test a single basket for cointegration and compute spread.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Price data with columns {symbol}_close
    basket : List[str]
        List of symbol names in the basket
        
    Returns:
    --------
    Optional[Dict]
        Dictionary with basket info, johansen results, eigenvector, spread, and log_prices
        Returns None if not cointegrated
    """
    # Extract basket prices
    basket_cols = [f'{sym}_close' for sym in basket]
    basket_prices = price_data[basket_cols].values
    
    # Convert to log prices
    log_prices = np.log(basket_prices)
    
    # Run Johansen test
    try:
        result = johansen_test(log_prices)
    except Exception as e:
        return None
    
    if not result['is_cointegrated']:
        return None
    
    # Compute spread using first eigenvector (normalized)
    eigenvector = result['eigenvectors'][:, 0]
    # Normalize so first element is 1 (standard normalization)
    eigenvector = eigenvector / eigenvector[0]
    
    spread = compute_spread(log_prices, eigenvector)
    
    return {
        'basket': basket,
        'johansen_result': result,
        'eigenvector': eigenvector,
        'spread': spread,
        'log_prices': log_prices
    }

