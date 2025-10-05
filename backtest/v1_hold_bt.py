#!/usr/bin/env python3
"""
Example: Run HODL strategy using the backtester.
- Ensures data exists (via backtester)
- Executes strategy and plots/prints summary
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import logging
import asyncio

# Add current directory to path (for local strategies package)
sys.path.append(str(Path(__file__).parent))
# Add src directory to path (for core engine modules)
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.tools import analyze_trades, plot_performance
from src.backtester import Backtester
from strategies.v1_hold import HoldStrategy
from src.hist_data import HistoricalDataCollector
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Main function to run complete backtest analysis"""
    print("Crypto Trading Strategy Backtester")
    print("=" * 50)
    
    # Resolve data dir at repo root
    hist_dir = Path(__file__).parents[2] / "historical_data"


    backtester = Backtester(
        historical_data_dir=str(hist_dir),
    )
    
    # 4. Run backtest and change strategies and start date/end date and timestep

    results = await backtester.run_backtest(strategy_class=HoldStrategy, 
    symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
    start_date=datetime.now(timezone.utc) - timedelta(days = 1), 
    end_date=datetime.now(timezone.utc), time_step=timedelta(hours=1))

    
    await backtester.print_results(results)

    # # Analyze trades
    # analyze_trades(results, "Hold Strategy")
    
    # Plot results
    plot_performance(results)
    
    # Save results
    results_summary = {
        'hold_strategy': results}

    data_dir = Path("historical_data")


if __name__ == "__main__":
    asyncio.run(main())
