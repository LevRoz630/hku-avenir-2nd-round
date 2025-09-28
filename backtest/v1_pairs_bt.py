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
# from strategies.v1_pairs_debug import PairTradingStrategy, set_pairs_config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # Configure pairs from the request
    pairs_config = [
        {
            'legs': ['APT-USDT', 'NEAR-USDT'],
            'use_futures': True,
            'lookback_days': 7,
            'entry_z': 1.5,
            'exit_z': 0.5,
            'max_alloc_frac': 0.5,
        },
        {
            'legs': ['OXT-USDT', 'ROSE-USDT'],
            'use_futures': True,
            'lookback_days': 3,
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
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()

    backtester = Backtester()
    strategy = PairTradingStrategy(symbols=base_symbols, historical_data_dir=str(hist_dir), lookback_days=50)
    results = backtester.run_backtest(
        strategy=strategy,
        symbols=base_symbols,
        start_date=start_date,
        end_date=end_date,
        time_step=timedelta(days = 1),
        market_type="futures"

    )

    backtester.print_results(results)

if __name__ == "__main__":
    main()


