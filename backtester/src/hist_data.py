#!/usr/bin/env python3
"""
Historical Data Collection Module

This module provides a comprehensive interface for collecting historical cryptocurrency
market data from Binance using the CCXT library. It supports spot and perpetual
futures markets with various data types including OHLCV, trades, funding rates, and
open interest data.

Classes
-------
HistoricalDataCollector
    Main class for collecting and managing historical cryptocurrency market data.

Examples
--------
>>> from datetime import datetime, timedelta
>>> from hist_data import HistoricalDataCollector
>>> 
>>> # Initialize collector
>>> collector = HistoricalDataCollector(data_dir="./data")
>>> 
>>> # Collect spot OHLCV data
>>> start_time = datetime.now() - timedelta(days=7)
>>> spot_data = collector.collect_spot_ohlcv("BTC-USDT", "1h", start_time, export=True)
>>> 
>>> # Collect perpetual mark price data
>>> mark_data = collector.collect_perpetual_mark_ohlcv("ETH-USDT", "15m", start_time, export=True)
>>> 
>>> # Use wrapper function for multiple data types
>>> end_time = datetime.now()
>>> data = collector.load_data_period("BTC-USDT", "1h", "ohlcv_spot", start_time, end_time, export=True)
"""

import pandas as pd
import time
from datetime import datetime, timedelta, timezone 
from pathlib import Path
import logging
from ccxt import binance
from ccxt.pro import binance as binance_pro

import re
from typing import Optional, Dict, Tuple
from src.utils import _is_utc, _get_number_of_periods, _get_timeframe_to_minutes
from src.utils import _convert_symbol_to_ccxt, _normalize_symbol_pair

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """
    Historical cryptocurrency market data collector using Binance API via CCXT.
    
    This class provides methods to collect various types of historical market data
    including OHLCV data, trades, funding rates, and open interest for both spot
    and perpetual futures markets. Data is automatically cached and can be exported
    to Parquet format.
    
    Parameters
    ----------
    data_dir : str, optional
        Directory path for storing collected data files. Default is "../hku-data/test_data".
        The directory will be created if it doesn't exist.
    
    Attributes
    ----------
    data_dir : Path
        Path object pointing to the data storage directory
    spot_ohlcv_data : dict
        Dictionary storing spot OHLCV data by symbol
    perpetual_mark_ohlcv_data : dict
        Dictionary storing perpetual mark price OHLCV data by symbol
    perpetual_index_ohlcv_data : dict
        Dictionary storing perpetual index price OHLCV data by symbol
    spot_trades_data : dict
        Dictionary storing spot trades data by symbol
    perpetual_trades_data : dict
        Dictionary storing perpetual trades data by symbol
    funding_rates_data : dict
        Dictionary storing funding rates data by symbol
    open_interest_data : dict
        Dictionary storing open interest data by symbol
    
    Examples
    --------
    >>> collector = HistoricalDataCollector("./my_data")
    >>> print(collector.data_dir)
    PosixPath('./my_data')
    """

    _FUTURES_DATA_TYPES = {
        'mark_ohlcv_futures',
        'index_ohlcv_futures',
        'funding_rates',
        'open_interest',
        'trades_futures',
    }
    
    def __init__(self, data_dir="../hku-data/test_data"):
        """
        Initialize historical data collector.
        
        Parameters
        ----------
        data_dir : str, optional
            Directory path for storing collected data files. Default is "../hku-data/test_data".
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage dictionaries
        self.spot_ohlcv_data = {}
        self.perpetual_mark_ohlcv_data = {}
        self.perpetual_index_ohlcv_data = {}
        self.spot_trades_data = {}
        self.perpetual_trades_data = {}
        self.funding_rates_data = {}
        self.open_interest_data = {}
        # Unified in-memory stores keyed by data_type for load/save convenience
        self.kind_map = {
            'ohlcv_spot': self.spot_ohlcv_data,
            'mark_ohlcv_futures': self.perpetual_mark_ohlcv_data,
            'index_ohlcv_futures': self.perpetual_index_ohlcv_data,
            'funding_rates': self.funding_rates_data,
            'open_interest': self.open_interest_data,
            'trades_futures': self.perpetual_trades_data,
        }
        
        try:
            # Initialize historical data exchanges (REST API)
            self.spot_exchange = binance({
                'sandbox': False,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })

            logger.info("Hist spot exchange initialized successfully")

            self.spot_exchange_pro = binance_pro({
                'sandbox': False,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })

            logger.info("Pro Binance spot exchange initialized successfully")

            self.futures_exchange_pro = binance_pro({
                'sandbox': False,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            logger.info("Pro Binance futures exchange initialized successfully")

            self.futures_exchange = binance({
                'sandbox': False,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            logger.info("Hist futures exchange initialized successfully")

        except Exception as e:
            logger.error(f"CCXT exchange initialization failed: {e}")
            self.spot_exchange = None
            self.spot_exchange_pro = None
            self.futures_exchange_pro = None
            self.futures_exchange = None
            raise ValueError(f"CCXT exchange initialization failed: {e}")

    def _normalize_symbol_input(self, symbol: str) -> str:
        """Canonicalize symbol formatting for downstream use."""
        if not symbol:
            raise ValueError("Symbol is required")
        normalized = str(symbol).strip().upper()
        return normalized.replace('/', '-').replace('_', '-')

    def _storage_key(self, symbol: str, data_type: str) -> str:
        """Return the dictionary/cache key for a given symbol/data_type pair."""
        normalized = self._normalize_symbol_input(symbol)
        if data_type == 'ohlcv_spot':
            normalized = normalized.replace('-PERP', '')
            if '-USDT' not in normalized:
                normalized = f"{normalized}-USDT"
            return normalized

        normalized = normalized.replace('-PERP', '').replace('-USDT', '')
        return normalized

    def _cache_token(self, symbol: str, data_type: str) -> str:
        """Token used inside cache filenames (underscored storage key)."""
        return self._storage_key(symbol, data_type).replace('-', '_')
    # ------------------------
    # Cache utilities
    # ------------------------
    def _cache_glob(self, kind: str, symbol: str, timeframe: str | None) -> list:
        cache_symbol = self._cache_token(symbol, kind)
        patterns = {
            'ohlcv_spot':           f"spot_{cache_symbol}_ohlcv_{timeframe}_*.parquet",
            'mark_ohlcv_futures':   f"perpetual_{cache_symbol}_mark_{timeframe}_*.parquet",
            'index_ohlcv_futures':  f"perpetual_{cache_symbol}_index_{timeframe}_*.parquet",
            'funding_rates':        f"perpetual_{cache_symbol}_funding_rates_*.parquet",
            'open_interest':        f"perpetual_{cache_symbol}_open_interest_{timeframe}_*.parquet",
            'trades_futures':       f"perpetual_{cache_symbol}_trades_*.parquet",
        }
        pattern = patterns[kind]
        files = sorted(self.data_dir.glob(pattern))

        if not files:
            legacy_symbol = f"{cache_symbol}_USDT"
            legacy_pattern = pattern.replace(cache_symbol, legacy_symbol)
            files = sorted(self.data_dir.glob(legacy_pattern))

        return files



    def load_data_period(self, symbol: str, timeframe: str, data_type: str, 
                        start_date: datetime, end_date: datetime,save_to_class: bool = False, load_from_class: bool = False, export: bool = False):
        """
        Unified wrapper function to load historical data for a specific time period.
        
        This is the main entry point for collecting historical data. It automatically
        determines the appropriate collection method based on the data type and handles
        all the complexity of data collection, caching, and filtering.
        
        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTC-USDT", "ETH-USDT")
        timeframe : str
            Timeframe for OHLCV data. Supported: '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'
        data_type : str
            Type of data to collect. Supported types:
            - "ohlcv_spot": Spot market OHLCV data
            - "index_ohlcv_futures": Perpetual futures index price OHLCV data
            - "mark_ohlcv_futures": Perpetual futures mark price OHLCV data
            - "funding_rates": Perpetual futures funding rates
            - "open_interest": Perpetual futures open interest
            - "trades_futures": Perpetual futures trades data
        start_date : datetime
            Start date and time for data collection
        end_date : datetime
            End date and time for data collection
        save_to_class : bool, optional
            Whether to save data to class stores. Default is False.
        load_from_class : bool, optional
            Whether to load data from class stores. Default is False.
            This is used for permutations as we shuffle the data inside of the class and using load_from_cache() leads to same data being used for algo
        export : bool, optional
            Whether to export data to Parquet file. Default is False.
        Returns
        -------
        pandas.DataFrame
            Filtered historical data for the specified time period
            
        Raises
        ------
        ValueError
            If start_date or end_date is None, or if start_date > end_date, or if data_type is invalid
        
        Examples
        --------
        >>> from datetime import datetime, timedelta
        >>> collector = HistoricalDataCollector()
        >>> 
        >>> # Collect spot OHLCV data
        >>> end_time = datetime.now()
        >>> start_time = end_time - timedelta(days=7)
        >>> spot_data = collector.load_data_period("BTC-USDT", "1h", "ohlcv_spot", 
        ...                                        start_time, end_time, save_to_class=True)
        >>> print(spot_data.head())
        
        >>> # Collect perpetual mark price data
        >>> mark_data = collector.load_data_period("ETH-USDT", "15m", "mark_ohlcv_futures",
        ...                                        start_time, end_time, save_to_class=True)
        >>> print(mark_data.columns)
        """
        data_types = ["ohlcv_spot", "index_ohlcv_futures", "mark_ohlcv_futures", 
                     "funding_rates", "open_interest", "trades_futures"]
        
        if start_date is None or end_date is None:
            raise ValueError("Start and end dates are required")

        if start_date > end_date:
            raise ValueError("Start date must be before end date")
        
        # check if start_date and end_date are in UTC timezone
        _is_utc(start_date)
        _is_utc(end_date)

        # Align to timeframe boundaries to avoid ms drift affecting cache hits
        minutes = _get_timeframe_to_minutes(timeframe)
        def _floor_to_tf(dt: datetime) -> datetime:
            # Floor to nearest timeframe boundary in UTC
            epoch_minutes = int((dt - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds() // 60)
            floored_minutes = (epoch_minutes // minutes) * minutes
            return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=floored_minutes)
        start_date = _floor_to_tf(start_date)
        # Make end inclusive by flooring as well; users expect stable end
        end_date = _floor_to_tf(end_date)

        if data_type not in data_types:
            raise ValueError(f"Invalid data type: {data_type}. Supported types: {data_types}")
        

        
        
        storage_symbol = self._storage_key(symbol, data_type)
        cache_timeframe = timeframe if data_type in ("ohlcv_spot", "mark_ohlcv_futures", "index_ohlcv_futures", "open_interest") else None

        # Cache-first, then collect
        try:
            if load_from_class:
                filtered_data = self.load_from_class(
                    data_type, symbol, start_date, end_date
                )
                if filtered_data is None or filtered_data.empty:
                    return None
                # Ensure we have a copy to avoid SettingWithCopyWarning
                filtered_data = filtered_data.copy()
            else:
                filtered_data = self.load_cached_window(
                    data_type,
                    symbol,
                    start_date,
                    end_date,
                    cache_timeframe
            )
            if filtered_data is None or filtered_data.empty:
                # During backtest execution, don't fetch from network for narrow time windows
                time_window_hours = (end_date - start_date).total_seconds() / 3600
                if time_window_hours < 2:  # Less than 2 hours = likely backtest price request
                    logger.warning(f"Cache miss during backtest: {data_type} {symbol} {timeframe} ({time_window_hours:.1f}h window)")
                    logger.warning("Data should be preloaded. Check initialization.")
                    return None

                logger.info(
                    f"Cache miss: {data_type} {symbol} {timeframe}, fetching from network and saving to cache"
                )

                if data_type == "mark_ohlcv_futures":
                    filtered_data = self.collect_perpetual_mark_ohlcv(storage_symbol, timeframe, start_date, export=export)
                elif data_type == "index_ohlcv_futures":
                    filtered_data = self.collect_perpetual_index_ohlcv(storage_symbol, timeframe, start_date, export=export)
                elif data_type == "ohlcv_spot":
                    filtered_data = self.collect_spot_ohlcv(storage_symbol, timeframe, start_date, export=export)
                elif data_type == "funding_rates":
                    filtered_data = self.collect_funding_rates(storage_symbol, start_date, export=export)
                elif data_type == "open_interest":
                    filtered_data = self.collect_open_interest(storage_symbol, timeframe, start_date, export=export)
                elif data_type == "trades_futures":
                    filtered_data = self.collect_perpetual_trades(storage_symbol, start_date, export=export)
                else:
                    raise ValueError(f"Invalid data type: {data_type}")
        
            if save_to_class:
                self.kind_map[data_type][storage_symbol] = filtered_data
                
        except Exception as e:
            logger.error(f"Error loading data: {e} for {symbol}")
            return None
        if filtered_data is None or filtered_data.empty:
            return None
        # Normalize DataFrame timestamps to tz-aware UTC (inputs already validated as UTC)
        filtered_data = filtered_data.copy()

        if not pd.api.types.is_datetime64_any_dtype(filtered_data["timestamp"]):
            filtered_data["timestamp"] = pd.to_datetime(filtered_data["timestamp"], errors='coerce', utc=True)
        else:
            ts = filtered_data["timestamp"]
            if ts.dt.tz is None:
                # Convert to UTC - use astype to ensure proper dtype
                localized_ts = ts.dt.tz_localize('UTC')
                filtered_data["timestamp"] = localized_ts.astype('datetime64[ns, UTC]')
            else:
                # Convert to UTC and ensure dtype compatibility
                converted_ts = ts.dt.tz_convert('UTC')
                filtered_data["timestamp"] = converted_ts.astype('datetime64[ns, UTC]')
        
        # Filter data to the requested time period (start_date/end_date already UTC)
        filtered_data = filtered_data[(filtered_data["timestamp"] >= start_date) & (filtered_data["timestamp"] <= end_date)]
        filtered_data["symbol"] = storage_symbol
        return filtered_data

    def load_from_class(self, kind: str, symbol: str, start_date, end_date):
        """Load a cached slice from in-memory stores for the requested window."""
        try:
            storage_symbol = self._storage_key(symbol, kind)
            store = self.kind_map.get(kind)
            if store is None:
                raise ValueError(f"Invalid kind: {kind}")
            df = store.get(storage_symbol)
            
            if df is None:
                return None


        except Exception as e:
            logger.error(f"Error loading data from class: {e}")
            return None
        
        try:
            s = pd.Timestamp(start_date).tz_convert('UTC') if pd.Timestamp(start_date).tzinfo is not None else pd.Timestamp(start_date, tz='UTC')
            e = pd.Timestamp(end_date).tz_convert('UTC') if pd.Timestamp(end_date).tzinfo is not None else pd.Timestamp(end_date, tz='UTC')
            # Work with a copy to avoid modifying the original DataFrame
            df = df.copy()
            ts = df['timestamp']

            if df['timestamp'].dt.tz is None:
                # Convert to UTC - use astype to ensure proper dtype
                localized_ts = df['timestamp'].dt.tz_localize('UTC')
                df['timestamp'] = localized_ts.astype('datetime64[ns, UTC]')
            else:
                # Convert to UTC and ensure dtype compatibility
                converted_ts = df['timestamp'].dt.tz_convert('UTC')
                df['timestamp'] = converted_ts.astype('datetime64[ns, UTC]')
            filtered = df[(df['timestamp'] >= s) & (df['timestamp'] <= e)]
            return filtered if not filtered.empty else None
        except Exception as e:
            logger.warning(f"Error loading data from class: {e}")
            return None

    def load_cached_window(self, kind: str, symbol: str, start_date, end_date, timeframe: str | None = None):
        """Return cached parquet data sliced to [start_date, end_date] if available, else None."""
        storage_symbol = self._storage_key(symbol, kind)
        files = self._cache_glob(kind, storage_symbol, timeframe)
        if not files:
            return None
        # Ensure tz-aware UTC timestamps for window bounds
        s = pd.Timestamp(start_date).tz_convert('UTC') if pd.Timestamp(start_date).tzinfo is not None else pd.Timestamp(start_date, tz='UTC')
        e = pd.Timestamp(end_date).tz_convert('UTC') if pd.Timestamp(end_date).tzinfo is not None else pd.Timestamp(end_date, tz='UTC')
        frames = []
        for fp in files:
            try:
                df = pd.read_parquet(fp)
                if 'timestamp' not in df.columns:
                    continue
                # Normalize timestamp dtype robustly (supports datetime, ms, s)
                ts = df['timestamp']
                if not pd.api.types.is_datetime64_any_dtype(ts):
                    try:
                        if pd.api.types.is_integer_dtype(ts):
                            vmax = int(pd.Series(ts).dropna().astype('int64').abs().max()) if len(ts) else 0
                            # Heuristics: 13-digit ~ ms, 10-digit ~ seconds
                            unit = 'ms' if vmax >= 10_000_000_000 else 's'
                            df['timestamp'] = pd.to_datetime(ts, unit=unit, errors='coerce')
                        else:
                            df['timestamp'] = pd.to_datetime(ts, errors='coerce')
                    except Exception:
                        df['timestamp'] = pd.to_datetime(ts, errors='coerce')
                # Force cached data timestamps to UTC timezone for consistent comparisons
                if df['timestamp'].dt.tz is None:
                    # Convert to UTC - use astype to ensure proper dtype
                    localized_ts = df['timestamp'].dt.tz_localize('UTC')
                    df['timestamp'] = localized_ts.astype('datetime64[ns, UTC]')
                else:
                    # Convert to UTC and ensure dtype compatibility
                    converted_ts = df['timestamp'].dt.tz_convert('UTC')
                    df['timestamp'] = converted_ts.astype('datetime64[ns, UTC]')
                df['symbol'] = storage_symbol
                ts_min, ts_max = df['timestamp'].min(), df['timestamp'].max()
                if ts_max < s or ts_min > e:
                    continue
                sliced = df[(df['timestamp'] >= s) & (df['timestamp'] <= e)]
                if sliced.empty:
                    continue
                frames.append(sliced)
            except Exception:
                continue
        if not frames:
            return None
        out = pd.concat(frames, ignore_index=True)

        if out.empty:
            return None
            
        return out.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    async def load_data_live(self, symbol: str, data_type: str):
        try:
            # Normalize symbols like "btc-usdt-perp" or "btc-usdt" to "BTC/USDT"
            watch_symbol = _normalize_symbol_pair(symbol)
            if watch_symbol is None:
                raise ValueError(f"Unable to normalize symbol: {symbol}")

            if data_type == "ohlcv_futures":
                df = await self.futures_exchange_pro.watch_ohlcv(watch_symbol)
                df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = symbol
                df['market_type'] = 'perpetual'
                await self.futures_exchange_pro.close()
                return df

            elif data_type == "ohlcv_spot":
                df = await self.spot_exchange_pro.watch_ohlcv(watch_symbol)
                df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = symbol
                df['market_type'] = 'spot'
                await self.spot_exchange_pro.close()
                return df
            else:
                raise ValueError(f"Invalid data type: {data_type}")

        except Exception as e:
            logger.error(f"Failed to load data live: {e}")
            return None
    
    def collect_spot_ohlcv(self, symbol: str, timeframe: str = '15m', 
                          start_time: datetime = None, end_time: datetime | None = None, export: bool = False):
        """
        Collect spot market OHLCV (Open, High, Low, Close, Volume) data.
        
        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTC-USDT", "ETH-USDT")
        timeframe : str, optional
            Timeframe for OHLCV data. Default is '15m'.
            Supported: '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'
        start_time : datetime
            Start time for data collection. Required.
        export : bool, optional
            Whether to export data to Parquet file. Default is False.
        
        Returns
        -------
        pandas.DataFrame or None
            OHLCV data with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'market_type']
            Returns None if no data is collected.
        
        Raises
        ------
        ValueError
            If start_time is None
        
        Examples
        --------
        >>> from datetime import datetime, timedelta
        >>> collector = HistoricalDataCollector()
        >>> 
        >>> start_time = datetime.now() - timedelta(days=7)
        >>> spot_data = collector.collect_spot_ohlcv("BTC-USDT", "1h", start_time, export=True)
        >>> print(spot_data.head())
        """
        if start_time is None:
            raise ValueError("Start time is required")

        storage_symbol = self._storage_key(symbol, 'ohlcv_spot')

        _is_utc(start_time)
        end_time = end_time or datetime.now(timezone.utc)
        _is_utc(end_time)
        max_records_per_request = 1000
        all_ohlcv = []
        # Generate safe filename
        start_str = start_time.strftime('%Y%m%d_%H%M%S') if isinstance(start_time, datetime) else str(start_time)
        end_str = end_time.strftime('%Y%m%d_%H%M%S') if isinstance(end_time, datetime) else str(end_time)

        
        cache_token = self._cache_token(storage_symbol, 'ohlcv_spot')
        filename = f"spot_{cache_token}_ohlcv_{timeframe}_{start_str}_{end_str}.parquet"

        logger.info(f"Collecting spot OHLCV for {storage_symbol}...")

        ccxt_symbol = _convert_symbol_to_ccxt(storage_symbol, "spot")

        # Collect data using unified collection method
        all_ohlcv = self._loop_data_collection(
            function=self.spot_exchange.fetch_ohlcv,
            ccxt_symbol=ccxt_symbol,
            timeframe=timeframe,
            limit=max_records_per_request,
            start_time=start_time,
            end_time=end_time,
            params=None,
            logger=logger
        )
        
        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = storage_symbol
            df['market_type'] = 'spot'
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            if export:
                df.to_parquet(self.data_dir / filename, index=False)
                logger.info(f"  Saved spot OHLCV for {storage_symbol}: {len(df):,} records to {filename}")
            self.spot_ohlcv_data[storage_symbol] = df
            return df
        else:
            logger.error(f"  No spot OHLCV data collected for {storage_symbol}")
            return None

    def collect_perpetual_mark_ohlcv(self, symbol: str, timeframe: str = '15m',
                                   start_time: datetime = None, end_time: datetime | None = None, export: bool = True):
        """
        Collect perpetual futures mark price OHLCV data.
        
        Mark price is the fair value price used for calculating unrealized P&L and
        margin requirements. It's calculated using the index price plus a premium/discount.
        
        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTC-USDT", "ETH-USDT")
        timeframe : str, optional
            Timeframe for OHLCV data. Default is '15m'.
            Supported: '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'
        start_time : datetime
            Start time for data collection. Required.
        export : bool, optional
            Whether to export data to Parquet file. Default is True.
        
        Returns
        -------
        pandas.DataFrame or None
            Mark price OHLCV data with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'market_type', 'price_type']
            Returns None if no data is collected.
        
        Raises
        ------
        ValueError
            If start_time is None
        
        Examples
        --------
        >>> from datetime import datetime, timedelta
        >>> collector = HistoricalDataCollector()
        >>> 
        >>> start_time = datetime.now() - timedelta(days=7)
        >>> mark_data = collector.collect_perpetual_mark_ohlcv("BTC-USDT", "1h", start_time, export=False)
        >>> print(mark_data.head())
        """

        if start_time is None:
            raise ValueError("Start time is required")

        storage_symbol = self._storage_key(symbol, 'mark_ohlcv_futures')

        max_records_per_request = 1000
        all_ohlcv = []

        end_time = end_time or datetime.now(timezone.utc)
        _is_utc(start_time)
        _is_utc(end_time)

        # Generate safe filename
        start_str = start_time.strftime('%Y%m%d') if isinstance(start_time, datetime) else str(start_time)
        end_str = end_time.strftime('%Y%m%d') if isinstance(end_time, datetime) else str(end_time)
        cache_token = self._cache_token(storage_symbol, 'mark_ohlcv_futures')
        filename = f"perpetual_{cache_token}_mark_{timeframe}_{start_str}_{end_str}.parquet"
        logger.info(f"Collecting perpetual mark price OHLCV for {storage_symbol}...")

        ccxt_symbol = _convert_symbol_to_ccxt(storage_symbol, "future")

        # Collect data using unified collection method with mark price parameter
        all_ohlcv = self._loop_data_collection(
            function=self.futures_exchange.fetch_ohlcv,
            ccxt_symbol=ccxt_symbol,
            timeframe=timeframe,
            limit=max_records_per_request,
            start_time=start_time,
            end_time=end_time,
            params={'price': 'mark'},
            logger=logger
        )
        
        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = storage_symbol
            df['market_type'] = 'perpetual'
            df['price_type'] = 'mark'
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            if export:
                df.to_parquet(self.data_dir / filename, index=False)
                logger.info(f"  Saved perpetual mark OHLCV for {storage_symbol}: {len(df):,} records to {filename}")
            return df
        else:
            logger.error(f"  No perpetual mark OHLCV data collected for {storage_symbol}")
            return None

    def collect_perpetual_index_ohlcv(self, symbol: str, timeframe: str = '15m',
                                    start_time: datetime = None, end_time: datetime | None = None, export: bool = True):
        """
        Collect perpetual futures index price OHLCV data.
        
        Index price is the underlying spot price reference used to calculate the mark price.
        It provides the spot market reference for fair value calculations.
        
        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTC-USDT", "ETH-USDT")
        timeframe : str, optional
            Timeframe for OHLCV data. Default is '15m'.
            Supported: '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'
        start_time : datetime
            Start time for data collection. Required.
        export : bool, optional
            Whether to export data to Parquet file. Default is True.
        
        Returns
        -------
        pandas.DataFrame or None
            Index price OHLCV data with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'market_type', 'price_type']
            Returns None if no data is collected.
        
        Raises
        ------
        ValueError
            If start_time is None
        
        Examples
        --------
        >>> from datetime import datetime, timedelta
        >>> collector = HistoricalDataCollector()
        >>> 
        >>> start_time = datetime.now() - timedelta(days=7)
        >>> index_data = collector.collect_perpetual_index_ohlcv("BTC-USDT", "1h", start_time, export=False)
        >>> print(index_data.head())
        """

        if start_time is None:
            raise ValueError("Start time is required")

        storage_symbol = self._storage_key(symbol, 'index_ohlcv_futures')

        max_records_per_request = 1000
        all_ohlcv = []

        end_time = end_time or datetime.now(timezone.utc)
        _is_utc(start_time)
        _is_utc(end_time)
        # Generate safe filename
        start_str = start_time.strftime('%Y%m%d_%H%M%S') if isinstance(start_time, datetime) else str(start_time)
        end_str = end_time.strftime('%Y%m%d_%H%M%S') if isinstance(end_time, datetime) else str(end_time)
        cache_token = self._cache_token(storage_symbol, 'index_ohlcv_futures')
        filename = f"perpetual_{cache_token}_index_{timeframe}_{start_str}_{end_str}.parquet"

        logger.info(f"Collecting perpetual index price OHLCV for {storage_symbol}...")

        ccxt_symbol = _convert_symbol_to_ccxt(storage_symbol, "future")

        # Collect data using unified collection method with index price parameter
        all_ohlcv = self._loop_data_collection(
            function=self.futures_exchange.fetch_ohlcv,
            ccxt_symbol=ccxt_symbol,
            timeframe=timeframe,
            limit=max_records_per_request,
            start_time=start_time,
            end_time=end_time,
            params={'price': 'index'},
            logger=logger
        )
        
        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = storage_symbol
            df['market_type'] = 'perpetual'
            df['price_type'] = 'index'
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            if export:
                df.to_parquet(self.data_dir / filename, index=False)
                logger.info(f"  Saved perpetual index OHLCV for {storage_symbol}: {len(df):,} records to {filename}")
            return df
        else:
            logger.error(f"  No perpetual index OHLCV data collected for {storage_symbol}")
            return None

    def collect_funding_rates(self, symbol: str, start_time: datetime = None, export: bool = True):
        """
        Collect perpetual futures funding rates data.
        
        Funding rates are periodic payments between long and short positions in perpetual
        futures contracts. They help keep the perpetual price close to the spot price.
        
        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTC-USDT", "ETH-USDT")
        start_time : datetime
            Start time for data collection. Required.
        export : bool, optional
            Whether to export data to Parquet file. Default is True.
        
        Returns
        -------
        pandas.DataFrame or None
            Funding rates data with columns: ['timestamp', 'funding_rate', 'funding_time', 'mark_price', 'index_price', 'symbol', 'market_type']
            Returns None if no data is collected.
        
        Raises
        ------
        ValueError
            If start_time is None
        
        Examples
        --------
        >>> from datetime import datetime, timedelta
        >>> collector = HistoricalDataCollector()
        >>> 
        >>> start_time = datetime.now() - timedelta(days=7)
        >>> funding_data = collector.collect_funding_rates("BTC-USDT", start_time, export=False)
        >>> print(funding_data.head())
        """
        if start_time is None:
            raise ValueError("Start time is required")

        storage_symbol = self._storage_key(symbol, 'funding_rates')

        end_time = datetime.now(timezone.utc)
        _is_utc(start_time)
        _is_utc(end_time)

        # Generate safe filename
        start_str = start_time.strftime('%Y%m%d_%H%M%S') if isinstance(start_time, datetime) else str(start_time)
        end_str = end_time.strftime('%Y%m%d_%H%M%S') if isinstance(end_time, datetime) else str(end_time)
        cache_token = self._cache_token(storage_symbol, 'funding_rates')
        filename = f"perpetual_{cache_token}_funding_rates_{start_str}_{end_str}.parquet"

        logger.info(f"Collecting funding rates for {storage_symbol}...")

        ccxt_symbol = _convert_symbol_to_ccxt(storage_symbol, "future")
        since = int(start_time.timestamp() * 1000)
        
        try:
            funding_rates = self.futures_exchange.fetch_funding_rate_history(
                ccxt_symbol, 
                since=since,
                limit=1000,
            )
            
            if funding_rates:
                processed_data = []
                for rate in funding_rates:
                    info = rate.get('info') or {}
                    processed_data.append({
                        'timestamp': rate.get('timestamp'),
                        'funding_rate': rate.get('fundingRate'),
                        'funding_time': rate.get('fundingTime') or info.get('fundingTime'),
                        'mark_price': rate.get('markPrice') or info.get('markPrice'),
                        'index_price': rate.get('indexPrice') or info.get('indexPrice'),
                        'symbol': storage_symbol,
                        'market_type': 'perpetual'
                    })
                
                df = pd.DataFrame(processed_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                # Coerce optional fields to proper types if present
                if 'funding_time' in df.columns:
                    df['funding_time'] = pd.to_datetime(df['funding_time'], unit='ms', errors='coerce')
                for _col in ['mark_price', 'index_price', 'funding_rate']:
                    if _col in df.columns:
                        df[_col] = pd.to_numeric(df[_col], errors='coerce')
                df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
                
                if export:
                    df.to_parquet(self.data_dir / filename, index=False)
                    logger.info(f"  Saved funding rates for {storage_symbol}: {len(df):,} records to {filename}")
                return df
            else:
                logger.error(f"  No funding rate data for {storage_symbol}")
                return None
            
        except Exception as e:
            logger.error(f"Failed to collect funding rates for {storage_symbol}: {e}")
            return None

    def collect_open_interest(self, symbol: str, timeframe: str = '15m',
                            start_time: datetime = None, export: bool = True):
        """
        Collect perpetual futures open interest data.
        
        Open interest represents the total number of outstanding derivative contracts
        that have not been settled. It's an important indicator of market activity.
        
        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTC-USDT", "ETH-USDT")
        timeframe : str, optional
            Timeframe for data collection. Default is '15m'.
            Supported: '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'
        start_time : datetime
            Start time for data collection. Required.
        export : bool, optional
            Whether to export data to Parquet file. Default is True.
        
        Returns
        -------
        pandas.DataFrame or None
            Open interest data with columns: ['timestamp', 'open_interest', 'open_interest_value', 'contract_type', 'symbol', 'market_type']
            Returns None if no data is collected.
        
        Raises
        ------
        ValueError
            If start_time is None
        
        Examples
        --------
        >>> from datetime import datetime, timedelta
        >>> collector = HistoricalDataCollector()
        >>> 
        >>> start_time = datetime.now() - timedelta(days=7)
        >>> oi_data = collector.collect_open_interest("BTC-USDT", "1h", start_time, export=False)
        >>> print(oi_data.head())
        """

        if start_time is None:
            raise ValueError("Start time is required")

        storage_symbol = self._storage_key(symbol, 'open_interest')

        max_records_per_request = 1000
        all_open_interest = []

        end_time = datetime.now(timezone.utc)
        _is_utc(start_time)
        _is_utc(end_time)

        # Generate safe filename
        start_str = start_time.strftime('%Y%m%d_%H%M%S') if isinstance(start_time, datetime) else str(start_time)
        end_str = end_time.strftime('%Y%m%d_%H%M%S') if isinstance(end_time, datetime) else str(end_time)
        cache_token = self._cache_token(storage_symbol, 'open_interest')
        filename = f"perpetual_{cache_token}_open_interest_{timeframe}_{start_str}_{end_str}.parquet"

        logger.info(f"Collecting open interest for {storage_symbol}...")

        ccxt_symbol = _convert_symbol_to_ccxt(storage_symbol, "future")

        # Collect data using unified collection method
        all_open_interest = self._loop_data_collection(
            function=self.futures_exchange.fetch_open_interest_history,
            ccxt_symbol=ccxt_symbol,
            timeframe=timeframe,
            limit=max_records_per_request,
            start_time=start_time,
            end_time=end_time,
            params=None,
            logger=logger
        )
        
        if all_open_interest:
            processed_data = []
            for oi in all_open_interest:
                processed_data.append({
                    'timestamp': oi.get('timestamp'),
                    'open_interest': oi.get('openInterestAmount'),
                    'open_interest_value': oi.get('openInterestValue'),
                    'contract_type': oi.get('contractType', 'perpetual'),
                    'symbol': storage_symbol,
                    'market_type': 'perpetual'
                })
            
            df = pd.DataFrame(processed_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            if export:
                df.to_parquet(self.data_dir / filename, index=False)
                logger.info(f"  Saved open interest for {storage_symbol}: {len(df):,} records to {filename}")
            return df
        else:
            logger.error(f"  No open interest data for {storage_symbol}")
            return None

    def collect_perpetual_trades(self, symbol: str, start_time: datetime = None, export: bool = True):
        """
        Collect perpetual futures trades data.
        
        Trades data contains individual trade executions including price, amount,
        side (buy/sell), and other trade metadata.
        
        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "BTC-USDT", "ETH-USDT")
        start_time : datetime
            Start time for data collection. Required.
        export : bool, optional
            Whether to export data to Parquet file. Default is True.
        
        Returns
        -------
        pandas.DataFrame or None
            Trades data with columns: ['timestamp', 'id', 'side', 'price', 'amount', 'cost', 'takerOrMaker', 'symbol', 'market_type']
            Returns None if no data is collected.
        
        Raises
        ------
        ValueError
            If start_time is None
        
        Examples
        --------
        >>> from datetime import datetime, timedelta
        >>> collector = HistoricalDataCollector()
        >>> 
        >>> start_time = datetime.now() - timedelta(days=7)
        >>> trades_data = collector.collect_perpetual_trades("BTC-USDT", start_time, export=False)
        >>> print(trades_data.head())
        """

        if start_time is None:
            raise ValueError("Start time is required")

        storage_symbol = self._storage_key(symbol, 'trades_futures')

        end_time = datetime.now(timezone.utc)
        _is_utc(start_time)
        _is_utc(end_time)

        # Generate safe filename
        start_str = start_time.strftime('%Y%m%d_%H%M%S') if isinstance(start_time, datetime) else str(start_time)
        end_str = end_time.strftime('%Y%m%d_%H%M%S') if isinstance(end_time, datetime) else str(end_time)
        cache_token = self._cache_token(storage_symbol, 'trades_futures')
        filename = f"perpetual_{cache_token}_trades_{start_str}_{end_str}.parquet"


        logger.info(f"Collecting perpetual trades for {storage_symbol}...")

        ccxt_symbol = _convert_symbol_to_ccxt(storage_symbol, "future")
        all_trades = []
        current_start_time = start_time
        
        while current_start_time < end_time:
            try:
                since = int(current_start_time.timestamp() * 1000)
                trades = self.futures_exchange.fetch_trades(ccxt_symbol, since=since, limit=1000)
                
                if trades:
                    all_trades.extend(trades)
                    logger.info(f"  Perpetual trades batch: {len(trades)} trades")
                    trades_time = trades[-1].get('timestamp')
                    current_start_time = datetime.fromtimestamp(trades_time / 1000, tz=timezone.utc)
                else:
                    break
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"  Error in perpetual trades batch: {e}")
                break
        
        if all_trades:
            processed_trades = []
            for trade in all_trades:
                processed_trades.append({
                    'timestamp': trade.get('timestamp'),
                    'id': trade.get('id'),
                    'side': trade.get('side'),
                    'price': trade.get('price'),
                    'amount': trade.get('amount'),
                    'cost': trade.get('cost'),
                    'takerOrMaker': trade.get('takerOrMaker'),
                    'symbol': storage_symbol,
                    'market_type': 'perpetual'
                })
            
            df = pd.DataFrame(processed_trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            if export:
                df.to_parquet(self.data_dir / filename, index=False)
                logger.info(f"  Saved perpetual trades for {storage_symbol}: {len(df):,} records to {filename}")
            self.perpetual_trades_data[storage_symbol] = df
            return df
        else:
            logger.error(f"  No perpetual trades data for {storage_symbol}")
            return None
  
    def _loop_data_collection(self, function, ccxt_symbol: str, timeframe: str, limit: int,
                            start_time: datetime, end_time: datetime, params=None, logger=None):
        """
        Generic data collection loop that handles different function signatures.
        
        This is a unified method for collecting data from different CCXT functions
        with proper batching, error handling, and rate limiting.
        
        Parameters
        ----------
        function : callable
            CCXT function to call (e.g., fetch_ohlcv, fetch_open_interest_history)
        ccxt_symbol : str
            Symbol in CCXT format (e.g., "BTC/USDT", "BTC/USDT:USDT")
        timeframe : str
            Timeframe for data collection
        limit : int
            Maximum number of records per request
        start_time : datetime
            Start time for data collection
        end_time : datetime
            End time for data collection
        params : dict, optional
            Additional parameters to pass to the function. Default is None.
        logger : logging.Logger, optional
            Logger instance for logging. Default is None.
        
        Returns
        -------
        list
            List of collected data records
        """
        _is_utc(start_time)
        _is_utc(end_time)
        # Calculate periods per day based on timeframe
        total_periods = _get_number_of_periods(timeframe, start_time, end_time)
        num_batches = (total_periods + limit - 1) // limit
        minutes = _get_timeframe_to_minutes(timeframe)
        all_data = []

        for batch in range(num_batches):
            try:
                batch_size = min(total_periods, limit)
                batch_start_time = end_time - batch_size * timedelta(minutes=minutes)
                since = int(batch_start_time.timestamp() * 1000)
                
                # Handle different function signatures
                if params:
                    if 'fetch_open_interest_history' in str(function):
                        # Open interest has different parameter order
                        data = function(ccxt_symbol, since=since, limit=limit, timeframe=timeframe)
                    else:
                        # OHLCV functions
                        data = function(ccxt_symbol, timeframe, since=since, limit=limit, params=params)
                else:
                    if 'fetch_open_interest_history' in str(function):
                        # Open interest has different parameter order
                        data = function(ccxt_symbol, since=since, limit=limit, timeframe=timeframe)
                    else:
                        # OHLCV functions
                        data = function(ccxt_symbol, timeframe, since=since, limit=limit)
                
                if data:
                    all_data.extend(data)
                    end_time = batch_start_time
                    if logger:
                        logger.info(f"  Data collection batch {batch+1}/{num_batches}: {batch_size} records")
                else:
                    break
                
                time.sleep(0.1)
                
            except Exception as e:
                if logger:
                    logger.error(f"  Error in data collection batch {batch+1}: {e}")
                time.sleep(1)
                continue
        
        return all_data