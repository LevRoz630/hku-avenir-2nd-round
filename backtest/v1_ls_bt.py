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
# from strategies.v1_pairs_debug import PairTradingStrategy, set_pairs_config4
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # Configure pairs from the request
    symbols = [
            'BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'ADA-USDT', 'XRP-USDT', 'DOGE-USDT'
        ]
    

    # Historical data directory
    hist_dir = Path(__file__).parents[2] / "hku-data" / "test_data"
    start_date = datetime.now(timezone.utc) - timedelta(days = 10)
    end_date = datetime.now(timezone.utc) - timedelta(days = 1)

    position_manager = V1LSPositionManager()
    backtester = Backtester()
    strategy = BTCAltShortStrategy(symbols=symbols, historical_data_dir=str(hist_dir), lookback_days=5)
    results = backtester.run_backtest(
        strategy=strategy,
        position_manager=position_manager,
        start_date=start_date,
        end_date=end_date,
        time_step=timedelta(hours = 1),
        market_type="futures",
    )
    backtester.print_results(results)
    backtester.plot_portfolio_value()
    backtester.plot_drawdown()
    backtester.plot_returns()
    # backtester.print_results(results)
    backtester.save_results(results, "v1_ls_bt")

    # results = backtester.run_permutation_backtest(
    #     strategy=strategy,
    #     position_manager=position_manager,
    #     start_date=start_date,
    #     end_date=end_date,
    #     time_step=timedelta(days = 1),
    #     market_type="futures",
    #     permutations=10,
    # )
    # print("p_value:", results.get("p_value"))
    # print("sharpes:", results.get("sharpes"))

if __name__ == "__main__":
    main()


