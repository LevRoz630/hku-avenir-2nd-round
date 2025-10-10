"""
Run hold strategy using the backtester as an example
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging

# Add current dir (for local strategies) and project src (engine)
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parents[2] / "src"))

from position_manager import PositionManager
from backtester import Backtester
from v1_hold import HoldStrategy
# from strategies.v1_pairs_debug import PairTradingStrategy, set_pairs_config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def main():
    """Main function to run complete backtest analysis"""
    print("Crypto Trading Strategy Backtester")
    print("=" * 50)
    
    # Resolve data dir at repo root
    hist_dir = Path(__file__).parents[2] / "historical_data"


    backtester = Backtester(
        historical_data_dir=str(hist_dir),
    )
    
    # 4. Run backtest and change strategies and start date/end date and timestep
    strategy = HoldStrategy(symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"], lookback_days = 0)
    position_manager = PositionManager()
    start_date = datetime.now(timezone.utc) - timedelta(days = 50)
    end_date = datetime.now(timezone.utc) - timedelta(days = 1)

    results = backtester.run_backtest(
        strategy=strategy,
        position_manager=position_manager,
        start_date=start_date,
        end_date=end_date,
        time_step=timedelta(days = 1),
        market_type="futures",
    )
    
    backtester.plot_positions(results)

    backtester.save_results(results, "hold_strategy")


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
