"""
Data loading module for cointegration testing.
Loads parquet files ONLY from hku-data directory (perpetual futures).
All CSV files are ignored - only .parquet files are processed.
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

logger = logging.getLogger(__name__)


def _load_single_parquet_file(args):
    """
    Worker function to load a single parquet file.
    Designed to be picklable for multiprocessing.
    """
    file_path, symbol, cutoff_date, price_column = args
    
    try:

        df = pd.read_parquet(file_path, engine='pyarrow')

        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' not in df.columns:
            if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            else:
                logger.warning(f"No timestamp found in {file_path.name}")
                return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Filter to cutoff date
        if cutoff_date:
            df = df[df['timestamp'] >= cutoff_date].copy()
        
        if len(df) == 0:
            return None
        
        # Use specified price column (close or mark)
        if price_column not in df.columns:
            logger.warning(f"Column {price_column} not found in {file_path.name}, using 'close'")
            price_column = 'close' if 'close' in df.columns else df.columns[1]
        
        df = df[['timestamp', price_column]].rename(columns={price_column: f'{symbol}_close'})
        df = df.set_index('timestamp').sort_index()
        
        return df
        
    except Exception as e:
        logger.warning(f"Failed to load {file_path.name}: {e}")
        return None


def load_price_data(data_dir: Path, symbols: Optional[List[str]] = None, 
                    years_back: Optional[float] = None,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    timeframe: str = '1h',
                    price_type: str = 'index',
                    max_workers: Optional[int] = None) -> pd.DataFrame:
    """
    Load price data from parquet files ONLY (perpetual futures).
    CSV files are ignored - only .parquet files matching the pattern are loaded.
    Returns aligned DataFrame with columns: {symbol}_close, ...
    
    Parameters:
    -----------
    data_dir : Path
        Directory containing parquet files (e.g., "../hku-data/test_data")
        Only .parquet files are processed; CSV files are ignored.
    symbols : Optional[List[str]]
        Optional list of symbols to filter (e.g., ["BTC", "ETH"])
        If None, discovers all available symbols from parquet files
    years_back : Optional[float]
        Number of years of historical data to load (if start_date not provided)
    start_date : Optional[datetime]
        Start date for data (UTC). If None, uses years_back
    end_date : Optional[datetime]
        End date for data (UTC). If None, uses all available data
    timeframe : str
        Timeframe to load: '1h' (index) or '15m' (mark)
    price_type : str
        'index' for index prices or 'mark' for mark prices
    max_workers : Optional[int]
        Number of parallel workers for file loading. If None, uses CPU count.
        
    Returns:
    --------
    pd.DataFrame
        Aligned price data with timestamp index, columns are {symbol}_close
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    # Determine file pattern based on timeframe and price_type
    if price_type == 'index':
        pattern = f"*_index_{timeframe}_*.parquet"
        price_column = 'close'
    elif price_type == 'mark':
        pattern = f"*_mark_{timeframe}_*.parquet"
        price_column = 'close'
    else:
        raise ValueError(f"Invalid price_type: {price_type}, must be 'index' or 'mark'")
    
    parquet_files = [f for f in data_dir.glob(pattern) if f.suffix == '.parquet']
    
    if not parquet_files:
        # Check if directory exists and has any files
        if not data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        all_files = list(data_dir.glob("*"))
        csv_files = [f for f in all_files if f.suffix == '.csv']
        other_parquet = [f for f in all_files if f.suffix == '.parquet']
        if csv_files:
            logger.warning(f"Found {len(csv_files)} CSV files in {data_dir} - these are ignored. Only parquet files are loaded.")
        if other_parquet and not parquet_files:
            raise ValueError(f"No parquet files found matching pattern '{pattern}' in {data_dir}. "
                           f"Found {len(other_parquet)} other parquet files. "
                           f"Expected pattern: perpetual_{{SYMBOL}}_{price_type}_{timeframe}_*.parquet")
        elif not other_parquet:
            raise ValueError(f"No parquet files found in {data_dir}. "
                           f"Expected pattern: perpetual_{{SYMBOL}}_{price_type}_{timeframe}_*.parquet")
    
    # Extract symbols from filenames if not provided
    if symbols:
        # Normalize symbols (remove -USDT, -PERP suffixes)
        normalized_symbols = [s.replace('-USDT', '').replace('-PERP', '') for s in symbols]
        parquet_files = [
            f for f in parquet_files 
            if any(norm_sym in f.stem for norm_sym in normalized_symbols)
        ]
    else:
        # Extract symbols from filenames
        symbols_set = set()
        for f in parquet_files:
            # Pattern: perpetual_{SYMBOL}_index_1h_...
            parts = f.stem.split('_')
            if len(parts) >= 2:
                symbols_set.add(parts[1])
        symbols = sorted(list(symbols_set))
    
    logger.info(f"Loading data from {len(parquet_files)} parquet files for {len(symbols)} symbols...")
    
    # Determine cutoff date
    cutoff_date = None
    if start_date:
        cutoff_date = start_date
    elif years_back:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=int(365 * years_back))
    
    # Prepare arguments for parallel loading
    # Ensure all files are parquet (double-check)
    args_list = []
    for parquet_file in parquet_files:
        if parquet_file.suffix != '.parquet':
            logger.warning(f"Skipping non-parquet file: {parquet_file.name}")
            continue
        # Extract symbol from filename
        parts = parquet_file.stem.split('_')
        symbol = parts[1] if len(parts) >= 2 else parquet_file.stem
        args_list.append((parquet_file, symbol, cutoff_date, price_column))
    
    # Load files in parallel
    all_data = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_load_single_parquet_file, args): args[0]
            for args in args_list
        }
        
        for future in as_completed(future_to_file):
            try:
                df = future.result()
                if df is not None and not df.empty:
                    symbol_col = df.columns[0]
                    symbol = symbol_col.replace('_close', '')
                    all_data[symbol] = df
            except Exception as e:
                file_path = future_to_file[future]
                logger.warning(f"Failed to load {file_path.name}: {e}")
    
    if not all_data:
        raise ValueError("No data loaded!")
    
    # Align all series on common timestamps
    logger.info("Aligning timestamps across all symbols...")
    combined = pd.concat(all_data.values(), axis=1, join='outer').sort_index()
    
    # Filter by end_date if provided
    if end_date:
        combined = combined[combined.index <= end_date]
    
    # Forward fill within day, then drop rows with >5% missing
    if timeframe == '15m':
        ffill_limit = 96  # Max 1 day forward fill (96 * 15min)
    else:
        ffill_limit = 24  # Max 1 day forward fill (24 * 1h)
    
    combined = combined.ffill(limit=ffill_limit)
    
    missing_pct = combined.isna().sum() / len(combined)
    valid_symbols = missing_pct[missing_pct < 0.05].index.tolist()
    combined = combined[valid_symbols]
    
    # Drop rows with any remaining NaN
    combined = combined.dropna()
    
    logger.info(f"Loaded {len(combined)} timestamps for {len(combined.columns)} symbols")
    logger.info(f"Date range: {combined.index.min()} to {combined.index.max()}")
    
    return combined

