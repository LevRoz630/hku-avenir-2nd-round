"""
Basket generation and initial cointegration screening.
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def _generate_combinations_from_cluster(args):
    """
    Worker function to generate combinations from a single cluster.
    Designed to be picklable for multiprocessing.
    """
    cluster_symbols, min_basket_size, max_basket_size = args

    if len(cluster_symbols) < 4:
        return []

    baskets = []
    for r in range(4, min(len(cluster_symbols) + 1, 7)):
        baskets.extend([list(combo) for combo in combinations(cluster_symbols, r)])

    return baskets


def generate_baskets_clustering(price_data: pd.DataFrame, n_clusters: int = 5,
                                max_workers: Optional[int] = None,
                                 max_combinations_per_cluster: Optional[int] = 20000) -> List[List[str]]:
    """
    Generate candidate baskets (4-6 symbols each) using hierarchical clustering on correlation matrix.
    Parallelizes combination generation across clusters.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Price data with columns {symbol}_close
    n_clusters : int
        Number of clusters for hierarchical clustering
    max_workers : Optional[int]
        Maximum number of worker processes. If None, uses CPU count.
    max_combinations_per_cluster : Optional[int]
        Maximum combinations to generate per cluster (default: 20000). If None, no limit.
        Helps prevent combinatorial explosion with large clusters.
        
    Returns:
    --------
    List[List[str]]
        List of candidate baskets, each basket is a list of symbol names
    """
    # Compute correlation matrix of log returns (more appropriate for multiplicative processes)
    # Log returns are statistically sound for financial time series that are multiplicative
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    corr_matrix = log_returns.corr().values
    
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
    
    cluster_list = list(clusters.values())
    
    # Filter clusters: if a cluster would generate too many combinations, limit its size
    filtered_clusters = []
    for cluster_symbols in cluster_list:
        if len(cluster_symbols) < 4:
            continue
        
        # Estimate total combinations (avoid computing all combinations)
        # Formula: C(n,2) + C(n,3) + ... + C(n,max) for n symbols
        # C(n,r) = n! / (r! * (n-r)!)
        n = len(cluster_symbols)
        total_combos = 0
        for r in range(4, min(n + 1, 7)):
            # Calculate C(n,r) efficiently
            if n > 50:  # Use approximation for very large n
                import math
                # Stirling approximation or simple upper bound
                # Upper bound: C(n,r) <= n^r / r!
                total_combos += int(n ** r / math.factorial(r))
            else:
                # Exact calculation for smaller n
                import math
                if r <= n:
                    total_combos += math.comb(n, r) if hasattr(math, 'comb') else int(
                        math.factorial(n) / (math.factorial(r) * math.factorial(n - r))
                    )
        
        # If too many combinations, limit cluster size intelligently
        if max_combinations_per_cluster and total_combos > max_combinations_per_cluster:
            # Use top N most correlated symbols within cluster
            cluster_cols = [col for col in price_data.columns 
                           if col.replace('_close', '') in cluster_symbols]
            if len(cluster_cols) > 0:
                cluster_returns = log_returns[cluster_cols]
                cluster_corr = cluster_returns.corr()
                # Get average correlation for each symbol
                avg_corr = cluster_corr.mean().sort_values(ascending=False)
                # Find max symbols that keep us under limit
                max_symbols = 4
                import math
                for n_test in range(4, len(cluster_symbols) + 1):
                    test_total = 0
                    for r in range(4, min(n_test + 1, 7)):
                        if hasattr(math, 'comb'):
                            test_total += math.comb(n_test, r)
                        else:
                            test_total += int(math.factorial(n_test) /
                                            (math.factorial(r) * math.factorial(n_test - r)))

                    if test_total <= max_combinations_per_cluster:
                        max_symbols = n_test
                    else:
                        break
                
                top_symbols = avg_corr.head(max_symbols).index.tolist()
                top_symbols = [s.replace('_close', '') for s in top_symbols]
                filtered_clusters.append(top_symbols)
                logger.info(f"Limited cluster size from {len(cluster_symbols)} to {len(top_symbols)} "
                           f"symbols (estimated {total_combos} -> ~{test_total} combinations)")
            else:
                # Fallback: just take first N symbols
                filtered_clusters.append(cluster_symbols[:min(20, len(cluster_symbols))])
        else:
            filtered_clusters.append(cluster_symbols)
    
    # Generate baskets from clusters (parallelized)
    args_list = [(cluster_symbols, 4, 6)  # Fixed min/max basket size for this evaluation
                 for cluster_symbols in filtered_clusters]

    baskets = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_cluster = {
            executor.submit(_generate_combinations_from_cluster, args): i
            for i, args in enumerate(args_list)
        }

        for future in as_completed(future_to_cluster):
            try:
                cluster_baskets = future.result()
                baskets.extend(cluster_baskets)
            except Exception as e:
                cluster_idx = future_to_cluster[future]
                logger.warning(f"Failed to generate baskets from cluster {cluster_idx}: {e}")

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
