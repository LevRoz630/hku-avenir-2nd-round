#!/usr/bin/env python3
"""Generate HMM regime labels from backtester price data."""

import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import glob

# Ensure local modules are importable
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent.parent / "hypothesis_testing" / "cointegration"))

from hmm_regimes import train_and_persist_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_price_data_from_backtester(data_dir: Path, price_type: str = "mark") -> pd.DataFrame:
    """Load price data into wide format with {SYMBOL}_close columns."""
    logger.info(f"Loading {price_type} price data from {data_dir}")

    # Find all parquet files for the specified price type
    pattern = f"perpetual_*_{price_type}_15m_*.parquet"
    price_files = list(data_dir.glob(pattern))

    if not price_files:
        raise FileNotFoundError(f"No {price_type} price files found in {data_dir}")

    logger.info(f"Found {len(price_files)} price data files")

    # Load and combine data
    price_frames = []

    for file_path in price_files:
        try:
            df = pd.read_parquet(file_path)

            # Extract symbol from filename (format: perpetual_{SYMBOL}_{price_type}_15m_...)
            filename = file_path.name
            symbol = filename.split('_')[1]  # Extract symbol from perpetual_{SYMBOL}_...

            # Ensure timestamp is datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            # Create wide format column name
            close_col = f"{symbol}_close"

            # Keep only close prices and rename column
            price_series = df[['close']].rename(columns={'close': close_col})

            price_frames.append(price_series)
            logger.debug(f"Loaded {symbol}: {len(price_series)} rows")

        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            continue

    if not price_frames:
        raise ValueError("No valid price data loaded")

    # Combine all symbols into wide format
    combined_data = pd.concat(price_frames, axis=1, join='outer')

    # Sort by timestamp and drop any rows with all NaN
    combined_data = combined_data.sort_index().dropna(how='all')

    logger.info(f"Combined data shape: {combined_data.shape}")
    logger.info(f"Time range: {combined_data.index.min()} to {combined_data.index.max()}")
    logger.info(f"Symbols loaded: {len([col for col in combined_data.columns if col.endswith('_close')])}")

    return combined_data

