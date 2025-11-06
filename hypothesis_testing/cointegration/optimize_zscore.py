"""
Optimize z-score parameters for trading: entry/exit thresholds and lookback windows.
Optimized for Sharpe ratio maximization.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from itertools import product

# Find workspace root
current = Path(__file__).resolve().parent
while current.name != 'hku-datawork' and current.parent != current:
    current = current.parent
workspace_root = current if current.name == 'hku-datawork' else Path('/workspace-hku/hku-datawork')
sys.path.insert(0, str(workspace_root))

from hypothesis_testing.cointegration.data_loader import load_price_data
from hypothesis_testing.cointegration.data_split import split_data_chronologically


def compute_zscore(spread: np.ndarray, lookback_bars: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rolling z-score for spread."""
    z_scores = np.full(len(spread), np.nan)
    
    for i in range(lookback_bars, len(spread)):
        window = spread[i - lookback_bars:i]
        mean = np.mean(window)
        std = np.std(window)
        if std > 0:
            z_scores[i] = (spread[i] - mean) / std
    
    valid_mask = ~np.isnan(z_scores)
    return z_scores, valid_mask


def simulate_trades(spread: np.ndarray, 
                    z_scores: np.ndarray,
                    entry_threshold: float,
                    exit_threshold: float,
                    bars_per_day: int) -> Dict:
    """
    Simulate mean reversion trades based on z-score signals.
    Strategy: Enter long when z < -entry_threshold, short when z > entry_threshold.
    Exit when z crosses exit_threshold.
    """
    positions = []
    position = None
    
    for i in range(len(z_scores)):
        if np.isnan(z_scores[i]):
            continue
            
        z = z_scores[i]
        
        # Exit conditions
        if position is not None:
            entry_idx, entry_z, side = position
            
            if side == 'long' and z >= exit_threshold:
                pnl = spread[i] - spread[entry_idx]
                positions.append((entry_idx, entry_z, side, i, z, pnl))
                position = None
            elif side == 'short' and z <= -exit_threshold:
                pnl = spread[entry_idx] - spread[i]
                positions.append((entry_idx, entry_z, side, i, z, pnl))
                position = None
        
        # Entry conditions
        if position is None:
            if z < -entry_threshold:
                position = (i, z, 'long')
            elif z > entry_threshold:
                position = (i, z, 'short')
    
    # Close open position
    if position is not None:
        entry_idx, entry_z, side = position
        pnl = (spread[-1] - spread[entry_idx]) if side == 'long' else (spread[entry_idx] - spread[-1])
        positions.append((entry_idx, entry_z, side, len(spread)-1, z_scores[-1], pnl))
    
    if not positions:
        return {
            'total_trades': 0,
            'sharpe_ratio': -np.inf,
            'total_return': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0
        }
    
    # Calculate statistics
    pnls = np.array([p[5] for p in positions])
    winning = pnls[pnls > 0]
    losing = pnls[pnls < 0]
    
    # Sharpe ratio: calculate from equity curve using actual test period
    if len(pnls) > 1:
        # Build equity curve (cumulative PnL)
        cumulative = np.cumsum(pnls)
        
        # Calculate total return over test period
        total_return = cumulative[-1]
        
        # Use a fixed notional for normalization (spread PnL is in log space)
        # Use max absolute cumulative as proxy for required capital
        max_abs_cumulative = np.max(np.abs(cumulative))
        notional = max(max_abs_cumulative, np.abs(total_return)) + 1e-10
        
        # Calculate period returns (per trade)
        returns = pnls / notional
        
        # Mean and std of returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Annualization: use actual test period length
        total_bars = positions[-1][3] - positions[0][0]
        total_days = total_bars / bars_per_day
        periods_per_year = 365.0 / (total_days / len(positions) + 1e-10)
        
        # Annualized Sharpe: sqrt(periods_per_year) * mean_return / std_return
        if std_return > 0 and np.isfinite(std_return) and np.isfinite(mean_return):
            sharpe = np.sqrt(periods_per_year) * mean_return / std_return
            # Validate sharpe
            if np.isnan(sharpe) or np.isinf(sharpe):
                sharpe = 0.0
        else:
            sharpe = 0.0
    else:
        sharpe = -np.inf
    
    # Max drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    max_drawdown = np.max(running_max - cumulative) if len(cumulative) > 0 else 0.0
    
    return {
        'total_trades': len(positions),
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'total_pnl': float(np.sum(pnls)),
        'avg_pnl': float(np.mean(pnls)),
        'win_rate': len(winning) / len(positions),
        'avg_win': float(np.mean(winning)) if len(winning) > 0 else 0.0,
        'avg_loss': float(np.mean(losing)) if len(losing) > 0 else 0.0,
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'total_return': float(cumulative[-1]) if len(cumulative) > 0 else 0.0,
        'avg_hold_days': float(np.mean([p[3] - p[0] for p in positions]) / bars_per_day)
    }


def optimize_zscore_params(basket_data: Dict,
                            test_data: pd.DataFrame,
                            entry_thresholds: List[float] = [1.0, 1.5, 2.0, 2.5, 3.0],
                            exit_thresholds: List[float] = [0.0, 0.5, 1.0],
                            lookback_days: List[int] = [7, 14, 21, 30, 45, 60],
                            bars_per_day: int = 96,
                            min_trades: int = 20) -> Dict:
    """
    Optimize z-score parameters for Sharpe ratio maximization.
    
    Parameters:
    -----------
    min_trades : int
        Minimum number of trades required (default: 20)
        Filters out parameter combinations with too few trades (statistically unreliable)
        
    Returns:
    --------
    dict with optimal parameters and all results sorted by Sharpe ratio
    """
    basket = basket_data['basket']
    eigenvector = np.array(basket_data['eigenvector'])
    
    # Extract basket prices and compute spread
    basket_cols = [f'{sym}_close' for sym in basket]
    if not all(col in test_data.columns for col in basket_cols):
        return {'error': 'Missing columns'}
    
    basket_prices = test_data[basket_cols].values
    
    # Validate prices
    if np.any(basket_prices <= 0):
        return {'error': 'Non-positive prices'}
    
    if np.any(np.isnan(basket_prices)) or np.any(np.isinf(basket_prices)):
        return {'error': 'NaN/Inf in prices'}
    
    log_prices = np.log(basket_prices)
    
    # Validate log prices
    if np.any(np.isnan(log_prices)) or np.any(np.isinf(log_prices)):
        return {'error': 'NaN/Inf in log prices'}
    
    # Validate eigenvector
    if np.any(np.isnan(eigenvector)) or np.any(np.isinf(eigenvector)):
        return {'error': 'NaN/Inf in eigenvector'}
    
    spread = log_prices @ eigenvector
    
    # Validate spread
    if np.any(np.isnan(spread)) or np.any(np.isinf(spread)):
        return {'error': 'NaN/Inf in spread'}
    
    # Test all parameter combinations
    results = []
    
    for entry_thresh, exit_thresh, lookback_d in product(entry_thresholds, exit_thresholds, lookback_days):
        lookback_bars = lookback_d * bars_per_day
        
        if lookback_bars >= len(spread):
            continue
        
        # Compute z-scores
        z_scores, valid_mask = compute_zscore(spread, lookback_bars)
        
        if not np.any(valid_mask):
            continue
        
        # Simulate trades
        trade_stats = simulate_trades(
            spread[valid_mask],
            z_scores[valid_mask],
            entry_thresh,
            exit_thresh,
            bars_per_day
        )
        
        # Skip if insufficient trades
        if trade_stats['total_trades'] < min_trades:
            continue
        
        # Add parameter info
        result = {
            'entry_threshold': entry_thresh,
            'exit_threshold': exit_thresh,
            'lookback_days': lookback_d,
            **trade_stats
        }
        results.append(result)
    
    if not results:
        return {'error': f'No parameter combinations with at least {min_trades} trades'}
    
    # Find optimal for Sharpe ratio
    best = max(results, key=lambda x: x['sharpe_ratio'])
    
    return {
        'basket': basket,
        'optimal_params': best,
        'all_results': sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True),
        'total_combinations_tested': len(results),
        'min_trades_filter': min_trades
    }


def main():
    """Main function to optimize z-score parameters for Sharpe ratio."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize z-score parameters for Sharpe ratio')
    parser.add_argument('--baskets-file', type=str, default='validated_baskets.json',
                       help='Path to validated baskets JSON file')
    parser.add_argument('--data-dir', type=str, default='/workspace-hku/hku-data/test_data',
                       help='Path to data directory')
    parser.add_argument('--min-trades', type=int, default=20,
                       help='Minimum number of trades required (default: 20)')
    parser.add_argument('--output', type=str, default='zscore_optimization.json',
                       help='Output file for results')
    args = parser.parse_args()
    
    # Load validated baskets
    baskets_file = Path(__file__).parent / args.baskets_file
    print(f"Loading baskets from {baskets_file}...")
    with open(baskets_file, 'r') as f:
        data = json.load(f)
    
    baskets = data['baskets']
    config = data['config']
    print(f"Found {len(baskets)} baskets\n")
    
    # Load and split data
    print(f"Loading price data...")
    price_data = load_price_data(
        data_dir=Path(args.data_dir),
        years_back=1.2,
        symbols=None,
        timeframe=config['timeframe'],
        price_type=config['price_type'],
        max_workers=None
    )
    
    splits = split_data_chronologically(
        price_data,
        cluster_split=(1.2, 0.8),
        cointegration_split=(0.8, 0.4),
        test_split=(0.4, 0.0)
    )
    test_data = splits['test_data']
    bars_per_day = 96 if config['timeframe'] == '15m' else 24
    
    print(f"Test data: {len(test_data)} samples\n")
    print(f"Optimizing z-score parameters for Sharpe ratio (min {args.min_trades} trades)...\n")
    
    # Optimize each basket
    optimization_results = []
    for i, basket_data in enumerate(baskets):
        print(f"Basket {i+1}/{len(baskets)}: {', '.join(basket_data['basket'])}")
        result = optimize_zscore_params(
            basket_data, 
            test_data, 
            bars_per_day=bars_per_day,
            min_trades=args.min_trades
        )
        
        if 'error' not in result:
            opt = result['optimal_params']
            print(f"  Optimal: entry={opt['entry_threshold']:.1f}σ, "
                  f"exit={opt['exit_threshold']:.1f}σ, "
                  f"lookback={opt['lookback_days']}d")
            print(f"  Sharpe={opt['sharpe_ratio']:.2f}, "
                  f"Return={opt['total_return']:.4f}, "
                  f"Win rate={opt['win_rate']:.1%}, "
                  f"Trades={opt['total_trades']}")
            optimization_results.append(result)
        else:
            print(f"  Error: {result['error']}")
        print()
    
    # Save results to JSON
    output_file = Path(__file__).parent / args.output
    
    # Prepare data for JSON serialization (convert numpy types)
    def convert_to_json_serializable(obj):
        """Convert numpy types to native Python types for JSON."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    
    serializable_results = convert_to_json_serializable(optimization_results)
    
    output_data = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'config': config,
        'min_trades_filter': args.min_trades,
        'objective': 'sharpe',
        'test_data_info': {
            'samples': len(test_data),
            'start_date': test_data.index.min().isoformat() if len(test_data) > 0 else None,
            'end_date': test_data.index.max().isoformat() if len(test_data) > 0 else None
        },
        'optimization_results': serializable_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {output_file}")
    print(f"  Optimized {len(optimization_results)} baskets")


if __name__ == "__main__":
    main()

