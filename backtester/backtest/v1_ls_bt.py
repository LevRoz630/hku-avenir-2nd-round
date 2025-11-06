#!/usr/bin/env python3
"""
Run pairs trading strategy using the backtester over the last ~90 days.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging

# Add current dir (for local strategies) and src (engine)
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

from position_managers.v1_ls_pm import V1LSPositionManager
from backtester import Backtester
from strategies.v1_ls import BTCAltShortStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # Configure pairs from the request
    symbols = [
            'BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'ADA-USDT', 'XRP-USDT', 'DOGE-USDT'
        ]
    

    # Historical data directory
    hist_dir = Path(__file__).parents[2] / "hku-data" / "test_data"
    start_date = datetime.now(timezone.utc) - timedelta(days = 100)
    end_date = datetime.now(timezone.utc) - timedelta(days = 1)

    position_manager = V1LSPositionManager()
    backtester = Backtester()
    strategy = BTCAltShortStrategy(symbols=symbols, historical_data_dir=str(hist_dir), lookback_days=5)
    print("Running single backtest first...")
    results = backtester.run_backtest(
        strategy=strategy,
        position_manager=position_manager,
        start_date=start_date,
        end_date=end_date,
        time_step=timedelta(hours = 1),
        market_type="futures",
    )
    backtester.print_results(results)
    
    print("\n" + "="*60)
    print("Running permutation backtest with 50 permutations...")
    print("="*60)
    perm_results = backtester.run_permutation_backtest(
        strategy=strategy,
        position_manager=position_manager,
        start_date=start_date,
        end_date=end_date,
        time_step=timedelta(hours = 1),
        market_type="futures",
        permutations=50,
    )
    
    print("\n" + "="*60)
    print("PERMUTATION TEST RESULTS")
    print("="*60)
    print(f"P-value: {perm_results.get('p_value', 'N/A'):.4f}")
    
    if perm_results.get('observed_results'):
        obs = perm_results['observed_results']
        print(f"\nObserved Run Results:")
        print(f"  Total Return: {obs.get('total_return', 0):.2%}")
        print(f"  Max Drawdown: {obs.get('max_drawdown', 0):.2%}")
        print(f"  Sharpe Ratio: {obs.get('sharpe_ratio', 0):.4f}")
        print(f"  Sortino Ratio: {obs.get('sortino_ratio', 0):.4f}")
        print(f"  Number of Trades: {len(obs.get('trade_history', []))}")
    
    sharpes = perm_results.get('sharpes', [])
    sortinos = perm_results.get('sortinos', [])
    if sharpes:
        print(f"\nPermutation Statistics:")
        print(f"  Observed Sharpe: {sharpes[0] if len(sharpes) > 0 else 'N/A':.4f}")
        print(f"  Mean Sharpe (permutations): {sum(sharpes[1:])/len(sharpes[1:]) if len(sharpes) > 1 else 'N/A':.4f}")
        print(f"  Min Sharpe: {min(sharpes[1:]) if len(sharpes) > 1 else 'N/A':.4f}")
        print(f"  Max Sharpe: {max(sharpes[1:]) if len(sharpes) > 1 else 'N/A':.4f}")
    
    backtester.save_results(results, "v1_ls_bt")

if __name__ == "__main__":
    main()


