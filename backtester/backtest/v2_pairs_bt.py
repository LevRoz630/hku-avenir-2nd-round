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
sys.path.append(str(Path(__file__).parent.parent))  # For 'from src.*' imports
sys.path.append(str(Path(__file__).parent.parent / "src"))  # For 'from utils', 'from hist_data' imports within src

from position_managers.v3_pairs_pm import PositionManager
from backtester import Backtester
from strategies.v2_pairs import PairTradingStrategy, set_pairs_config
# from strategies.v1_pairs_debug import PairTradingStrategy, set_pairs_config4

# Configure logging to both console and file
def setup_logging():
    """Setup logging to both console and timestamped log file."""
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"backtest_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to file: {log_file}")
    
    return log_file

# Setup logging before creating other loggers
log_file = setup_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info(f"{'='*80}")
    logger.info(f"Starting backtest run at {datetime.now(timezone.utc)}")
    logger.info(f"{'='*80}")
    
    # Configure pairs from the request
    pairs_config = [
        {
            'legs': ['APT-USDT', 'NEAR-USDT'],
            'use_futures': True,
            'entry_z': 1.5,
            'exit_z': 0.5,
            'max_alloc_frac': 0.5,
        },
        {
            'legs': ['OXT-USDT', 'ROSE-USDT'],
            'use_futures': True,
            'entry_z': 1.5,
            'exit_z': 0.5,
            'max_alloc_frac': 0.5,
        },
    ]

    set_pairs_config(pairs_config)

    # Build the base symbols list for data loading (legs already include -USDT)
    base_symbols = []
    for cfg in pairs_config:
        for base in cfg['legs']:
            sym = base
            if sym not in base_symbols:
                base_symbols.append(sym)

    # Historical data directory
    hist_dir = Path(__file__).parents[2] / "hku-data" / "test_data"
    start_date = datetime.now(timezone.utc) - timedelta(days = 10)
    end_date = datetime.now(timezone.utc) - timedelta(days = 4)
    
    logger.info(f"Backtest period: {start_date} to {end_date}")
    logger.info(f"Pairs: {[cfg['legs'] for cfg in pairs_config]}")

    position_manager = PositionManager(
        portfolio_alloc_frac=0.8,
        risk_method='min_volatility',
        rebalance_threshold=0.15,  # 15% drift
        pairs_config=pairs_config
    )
    backtester = Backtester()
    strategy = PairTradingStrategy(symbols=base_symbols, historical_data_dir=str(hist_dir), lookback_days=90)
    
    results = backtester.run_backtest(
        strategy=strategy,
        position_manager=position_manager,
        start_date=start_date,
        end_date=end_date,
        time_step=timedelta(days = 1),
        market_type="futures",
    )
    
    logger.info(f"{'='*80}")
    logger.info(f"Backtest completed. Log saved to: {log_file}")
    logger.info(f"{'='*80}")
    # backtester.print_results(results)
    # backtester.save_results(results, "v1_pairs_bt")
    backtester.plot_portfolio_value()
    backtester.plot_drawdown()
    backtester.plot_returns()
    backtester.plot_positions()

    # results = backtester.run_permutation_backtest(
    #     strategy=strategy,
    #     position_manager=position_manager,
    #     start_date=start_date,
    #     end_date=end_date,
    #     time_step=timedelta(days = 1),
    #     market_type="futures",
    #     permutations=100,
    # )
    # print("p_value:", results.get("p_value"))
    # print("sharpes:", results.get("sharpes"))

if __name__ == "__main__":
    main()


