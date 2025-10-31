"""
Benchmark script to compare sequential vs parallel cointegration testing.
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# Add workspace root to path
workspace_root = Path('/workspace-hku/hku-datawork')
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from hypothesis_testing.cointegration import (
    test_baskets_cointegration_parallel,
    johansen_test
)
from hypothesis_testing.cointegration.basket_generator import compute_spread as _compute_spread


def test_baskets_sequential(price_data: pd.DataFrame, candidate_baskets):
    """Sequential version for comparison."""
    cointegrated_baskets = []
    for basket in candidate_baskets:
        try:
            basket_cols = [f'{sym}_close' for sym in basket]
            basket_prices = price_data[basket_cols].values
            log_prices = np.log(basket_prices)
            result = johansen_test(log_prices)
            if result['is_cointegrated']:
                eigenvector = result['eigenvectors'][:, 0]
                eigenvector = eigenvector / eigenvector[0]
                spread = _compute_spread(log_prices, eigenvector)
                cointegrated_baskets.append({
                    'basket': basket,
                    'johansen_result': result,
                    'eigenvector': eigenvector,
                    'spread': spread,
                    'log_prices': log_prices
                })
        except:
            continue
    return cointegrated_baskets


def create_synthetic_data(n_symbols=50, n_periods=10000):
    """Create synthetic price data for benchmarking."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n_periods, freq='15min', tz='UTC')
    
    # Generate correlated price series
    base_trend = np.cumsum(np.random.randn(n_periods) * 0.001)
    prices = {}
    for i in range(n_symbols):
        # Each symbol follows base trend with some noise and correlation
        noise = np.random.randn(n_periods) * 0.01
        correlation = 0.3 + np.random.rand() * 0.4  # 0.3-0.7 correlation
        symbol_prices = base_trend * correlation + noise
        symbol_prices = np.exp(np.cumsum(symbol_prices))  # Convert to prices
        prices[f'SYM{i}_close'] = symbol_prices
    
    return pd.DataFrame(prices, index=dates)


def benchmark():
    """Run benchmark comparison."""
    print("="*60)
    print("BENCHMARK: Sequential vs Parallel Cointegration Testing")
    print("="*60)
    
    # Create synthetic data
    print("\n1. Creating synthetic data...")
    price_data = create_synthetic_data(n_symbols=50, n_periods=10000)
    print(f"   Created {len(price_data)} timestamps, {len(price_data.columns)} symbols")
    
    # Generate test baskets (simple pairs and triplets)
    print("\n2. Generating test baskets...")
    symbols = [col.replace('_close', '') for col in price_data.columns]
    from itertools import combinations
    test_baskets = []
    # Add pairs
    test_baskets.extend([list(c) for c in combinations(symbols[:30], 2)])  # ~435 pairs
    # Add triplets
    test_baskets.extend([list(c) for c in combinations(symbols[:20], 3)])  # ~1140 triplets
    test_baskets = test_baskets[:500]  # Limit to 500 for quick test
    print(f"   Testing {len(test_baskets)} baskets")
    
    # Warmup
    print("\n3. Warmup run...")
    _ = test_baskets_cointegration_parallel(price_data, test_baskets[:10], max_workers=2, batch_size=5)
    
    # Benchmark 1: Sequential
    print("\n4. Benchmark: Sequential")
    start = time.time()
    sequential_results = test_baskets_sequential(price_data, test_baskets)
    sequential_time = time.time() - start
    print(f"   Time: {sequential_time:.2f}s")
    print(f"   Found: {len(sequential_results)} cointegrated baskets")
    
    # Benchmark 2: Parallel (old - no batching, batch_size=1)
    print("\n5. Benchmark: Parallel (no batching, batch_size=1)")
    start = time.time()
    parallel_no_batch_results = test_baskets_cointegration_parallel(
        price_data, test_baskets, max_workers=4, batch_size=1
    )
    parallel_no_batch_time = time.time() - start
    print(f"   Time: {parallel_no_batch_time:.2f}s")
    print(f"   Found: {len(parallel_no_batch_results)} cointegrated baskets")
    print(f"   Speedup vs sequential: {sequential_time / parallel_no_batch_time:.2f}x")
    
    # Benchmark 3: Parallel (batched, batch_size=100)
    print("\n6. Benchmark: Parallel (batched, batch_size=100)")
    start = time.time()
    parallel_batched_results = test_baskets_cointegration_parallel(
        price_data, test_baskets, max_workers=4, batch_size=100
    )
    parallel_batched_time = time.time() - start
    print(f"   Time: {parallel_batched_time:.2f}s")
    print(f"   Found: {len(parallel_batched_results)} cointegrated baskets")
    print(f"   Speedup vs sequential: {sequential_time / parallel_batched_time:.2f}x")
    print(f"   Speedup vs no-batch parallel: {parallel_no_batch_time / parallel_batched_time:.2f}x")
    
    # Benchmark 4: Parallel (batched, batch_size=50)
    print("\n7. Benchmark: Parallel (batched, batch_size=50)")
    start = time.time()
    parallel_batched_50_results = test_baskets_cointegration_parallel(
        price_data, test_baskets, max_workers=4, batch_size=50
    )
    parallel_batched_50_time = time.time() - start
    print(f"   Time: {parallel_batched_50_time:.2f}s")
    print(f"   Found: {len(parallel_batched_50_results)} cointegrated baskets")
    print(f"   Speedup vs sequential: {sequential_time / parallel_batched_50_time:.2f}x")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Sequential:           {sequential_time:.2f}s")
    print(f"Parallel (no batch):  {parallel_no_batch_time:.2f}s ({sequential_time/parallel_no_batch_time:.2f}x)")
    print(f"Parallel (batch=50): {parallel_batched_50_time:.2f}s ({sequential_time/parallel_batched_50_time:.2f}x)")
    print(f"Parallel (batch=100): {parallel_batched_time:.2f}s ({sequential_time/parallel_batched_time:.2f}x)")
    print("="*60)
    
    # Verify results are same
    sequential_baskets = {tuple(sorted(r['basket'])) for r in sequential_results}
    parallel_baskets = {tuple(sorted(r['basket'])) for r in parallel_batched_results}
    if sequential_baskets == parallel_baskets:
        print("✓ Results match between sequential and parallel!")
    else:
        print(f"⚠ Results differ: sequential={len(sequential_baskets)}, parallel={len(parallel_baskets)}")


if __name__ == "__main__":
    benchmark()

