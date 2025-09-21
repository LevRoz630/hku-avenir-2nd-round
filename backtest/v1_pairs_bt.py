#!/usr/bin/env python3
"""
Run pairs trading strategy using the backtester over the last ~90 days.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add current dir (for local strategies) and src (engine)
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

from backtester import Backtester
from strategies.v1_pairs import PairTradingStrategy, set_pairs_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # Configure pairs from the request
    pairs_config = [
        {
            'legs': ['APT', 'NEAR'],
            'use_futures': True,
            'lookback_days': 7,
            'entry_z': 0.326,
            'exit_z': 0.559,
            'max_alloc_frac': 0.5,
        },
        {
            'legs': ['OXT', 'ROSE'],
            'use_futures': True,
            'lookback_days': 3,
            'entry_z': 0.333,
            'exit_z': 0.455,
            'max_alloc_frac': 0.5,
        },
    ]

    set_pairs_config(pairs_config)

    # Build the base symbols list for data loading
    base_symbols = []
    for cfg in pairs_config:
        for base in cfg['legs']:
            sym = f"{base}-USDT"
            if sym not in base_symbols:
                base_symbols.append(sym)

    # Historical data directory at repo root
    hist_dir = Path(__file__).parents[2] / "historical_data"

    backtester = Backtester(historical_data_dir=str(hist_dir))

    start_date = datetime.now() - timedelta(days=3)
    end_date = datetime.now()

    results = backtester.run_backtest(
        strategy_class=PairTradingStrategy,
        symbols=base_symbols,
        start_date=start_date,
        end_date=end_date,
        time_step=timedelta(minutes=15),
    )

    backtester.print_results(results)

if __name__ == "__main__":
    main()


