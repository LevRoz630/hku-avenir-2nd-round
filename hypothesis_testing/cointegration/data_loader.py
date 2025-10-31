"""
Data loading module for cointegration testing.
Reads CSV files directly from hku-data directory.
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_price_data(data_dir: Path, symbols: Optional[List[str]] = None, 
                    years_back: float = 1.0) -> pd.DataFrame:
    """
    Load last N years of 15-minute data from CSV files.
    Returns aligned DataFrame with columns: {symbol}_close, ...
    
    Parameters:
    -----------
    data_dir : Path
        Directory containing CSV files (e.g., "../hku-data/test_data")
    symbols : Optional[List[str]]
        Optional list of symbols to filter (e.g., ["BTC-USDT-PERP", "ETH-USDT-PERP"])
        If None, discovers all available symbols from CSV files
    years_back : float
        Number of years of historical data to load
        
    Returns:
    --------
    pd.DataFrame
        Aligned price data with timestamp index, columns are {symbol}_close
    """
    csv_files = list(data_dir.glob("*_15m_*.csv"))
    
    if symbols:
        # Filter to requested symbols
        csv_files = [f for f in csv_files if any(sym in f.stem for sym in symbols)]
    
    logger.info(f"Loading data from {len(csv_files)} CSV files...")
    
    # Determine cutoff date
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=int(365 * years_back))
    
    all_data = {}
    for csv_file in tqdm(csv_files, desc="Loading CSVs"):
        try:
            # Extract symbol from filename (e.g., "ADA-USDT-PERP_15m_1460d.csv" -> "ADA-USDT-PERP")
            symbol = csv_file.stem.split('_')[0]
            
            # Load CSV
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Filter to last N years
            df = df[df['timestamp'] >= cutoff_date].copy()
            
            if len(df) == 0:
                continue
            
            # Use close price
            df = df[['timestamp', 'close']].rename(columns={'close': f'{symbol}_close'})
            df = df.set_index('timestamp').sort_index()
            
            all_data[symbol] = df
            
        except Exception as e:
            logger.warning(f"Failed to load {csv_file.name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data loaded!")
    
    # Align all series on common timestamps
    logger.info("Aligning timestamps across all symbols...")
    combined = pd.concat(all_data.values(), axis=1, join='outer').sort_index()
    
    # Forward fill within day, then drop rows with >5% missing
    combined = combined.ffill(limit=96)  # Max 1 day forward fill (96 * 15min)
    
    missing_pct = combined.isna().sum() / len(combined)
    valid_symbols = missing_pct[missing_pct < 0.05].index.tolist()
    combined = combined[valid_symbols]
    
    # Drop rows with any remaining NaN
    combined = combined.dropna()
    
    logger.info(f"Loaded {len(combined)} timestamps for {len(combined.columns)} symbols")
    logger.info(f"Date range: {combined.index.min()} to {combined.index.max()}")
    
    return combined

