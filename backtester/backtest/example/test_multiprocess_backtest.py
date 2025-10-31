#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "position_managers"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from v1_hold import HoldStrategy
    from example import PositionManager
    from backtester import Backtester
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("Testing Multiprocessing Permutation Backtest")
    print("=" * 60)
    
    # Use a short time period and few permutations for quick testing
    hist_dir = Path(__file__).parent.parent.parent / "hku-data" / "test_data"
    hist_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple strategy with 2 symbols
    symbols = ["BTC-USDT", "ETH-USDT"]
    strategy = HoldStrategy(symbols=symbols, lookback_days=0)
    position_manager = PositionManager()
    
    # Short date range for quick testing
    start_date = datetime.now(timezone.utc) - timedelta(days=10)
    end_date = datetime.now(timezone.utc) - timedelta(days=1)
    
    backtester = Backtester(historical_data_dir=str(hist_dir))
    
    print(f"\nRunning permutation backtest:")
    print(f"  Symbols: {symbols}")
    print(f"  Date range: {start_date.date()} to {end_date.date()}")
    print(f"  Permutations: 5 (including 1 observed)")
    print(f"  Workers: 2")
    print(f"  Note: This will download data if not already cached\n")
    
    try:
        # First run a simple backtest to ensure data is loaded
        print("Running initial backtest to load data...")
        single_result = backtester.run_backtest(
            strategy=strategy,
            position_manager=position_manager,
            start_date=start_date,
            end_date=end_date,
            time_step=timedelta(hours=6),
            market_type="futures",
        )
        print(f"Initial backtest completed. Portfolio value: {single_result.get('final_balance', 'N/A')}\n")
        
        # Now run permutation test
        print("Running multiprocessing permutation backtest...")
        results = backtester.run_permutation_backtest(
            strategy=strategy,
            position_manager=position_manager,
            start_date=start_date,
            end_date=end_date,
            time_step=timedelta(hours=6),  # 6-hour steps for quick test
            market_type="futures",
            permutations=4,  # 4 shuffled + 1 observed = 5 total
            max_workers=2  # Use 2 workers for testing
        )
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"P-value: {results.get('p_value', 'N/A'):.4f}")
        
        if results.get('observed_results'):
            obs = results['observed_results']
            print(f"\nObserved Run Results:")
            print(f"  Total Return: {obs.get('total_return', 0):.2%}")
            print(f"  Max Drawdown: {obs.get('max_drawdown', 0):.2%}")
            print(f"  Sharpe Ratio: {obs.get('sharpe_ratio', 0):.4f}")
            print(f"  Sortino Ratio: {obs.get('sortino_ratio', 0):.4f}")
            print(f"  Number of Trades: {len(obs.get('trade_history', []))}")
        
        sharpes = results.get('sharpes', [])
        sortinos = results.get('sortinos', [])
        print(f"\nPermutation Statistics:")
        print(f"  Sharpe ratios computed: {len(sharpes)}")
        print(f"  Sortino ratios computed: {len(sortinos)}")
        if sharpes:
            print(f"  Mean Sharpe (permutations): {sum(sharpes)/len(sharpes):.4f}")
        if sortinos:
            print(f"  Mean Sortino (permutations): {sum(sortinos)/len(sortinos):.4f}")
        
        print("\n" + "=" * 60)
        print("TEST PASSED: Multiprocessing permutation backtest completed successfully!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

