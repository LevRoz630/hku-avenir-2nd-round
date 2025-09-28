#!/usr/bin/env python3
"""
Historical Data Collection Module

This module provides a comprehensive interface for collecting historical cryptocurrency
market data from Binance using the CCXT library. It supports both spot and perpetual
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
from datetime import datetime, timedelta
from pathlib import Path
import logging
from ccxt import binance as binance_sync
from ccxt.pro import binance as binance_pro
import re
from typing import Optional, Dict, Tuple

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
        
        # Initialize live data exchanges (WebSocket)
        self.live_futures_exchange = binance_pro({
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        self.live_spot_exchange = binance_pro({
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # Initialize historical data exchanges (REST API)
        self.spot_exchange = binance_sync({
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        self.futures_exchange = binance_sync({
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

    async def close(self):
        """Close underlying exchange clients (websocket and REST)."""
        try:
            if hasattr(self, 'live_futures_exchange') and self.live_futures_exchange is not None:
                await self.live_futures_exchange.close()
        except Exception:
            pass
        try:
            if hasattr(self, 'live_spot_exchange') and self.live_spot_exchange is not None:
                await self.live_spot_exchange.close()
        except Exception:
            pass
        try:
            if hasattr(self, 'spot_exchange') and self.spot_exchange is not None and hasattr(self.spot_exchange, 'close'):
                self.spot_exchange.close()
        except Exception:
            pass
        try:
            if hasattr(self, 'futures_exchange') and self.futures_exchange is not None and hasattr(self.futures_exchange, 'close'):
                self.futures_exchange.close()
        except Exception:
            pass

    def load_data_period(self, symbol: str, timeframe: str, data_type: str, 
                        start_date: datetime, end_date: datetime, export: bool = False):
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
        export : bool, optional
            Whether to export data to Parquet files. Default is False.
        
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
        ...                                        start_time, end_time, export=True)
        >>> print(spot_data.head())
        
        >>> # Collect perpetual mark price data
        >>> mark_data = collector.load_data_period("ETH-USDT", "15m", "mark_ohlcv_futures",
        ...                                        start_time, end_time, export=True)
        >>> print(mark_data.columns)
        """
        data_types = ["ohlcv_spot", "index_ohlcv_futures", "mark_ohlcv_futures", 
                     "funding_rates", "open_interest", "trades_futures"]
        
        if start_date is None or end_date is None:
            raise ValueError("Start and end dates are required")
        if start_date > end_date:
            raise ValueError("Start date must be before end date")
        
        if data_type not in data_types:
            raise ValueError(f"Invalid data type: {data_type}. Supported types: {data_types}")
        
        # Route to appropriate collection method based on data type
        if data_type == "mark_ohlcv_futures":
            self.collect_perpetual_mark_ohlcv(symbol, timeframe, start_date, export=export)
            filtered_data = self.perpetual_mark_ohlcv_data[symbol]
        elif data_type == "index_ohlcv_futures":
            self.collect_perpetual_index_ohlcv(symbol, timeframe, start_date, export=export)
            filtered_data = self.perpetual_index_ohlcv_data[symbol]
        elif data_type == "ohlcv_spot":
            self.collect_spot_ohlcv(symbol, timeframe, start_date, export=export)
            filtered_data = self.spot_ohlcv_data[symbol]
        elif data_type == "funding_rates":
            self.collect_funding_rates(symbol, start_date, export=export)
            filtered_data = self.funding_rates_data[symbol]
        elif data_type == "open_interest":
            self.collect_open_interest(symbol, timeframe, start_date, export=export)
            filtered_data = self.open_interest_data[symbol]
        elif data_type == "trades_futures":
            self.collect_perpetual_trades(symbol, start_date, export=export)
            filtered_data = self.perpetual_trades_data[symbol]
        else:
            raise ValueError(f"Invalid data type: {data_type}")

        # Filter data to the requested time period
        filtered_data = filtered_data[(filtered_data["timestamp"] >= start_date) & 
                                    (filtered_data["timestamp"] <= end_date)]
        return filtered_data


    def collect_spot_ohlcv(self, symbol: str, timeframe: str = '15m', 
                          start_time: datetime = None, export: bool = False):
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
        max_records_per_request = 1000
        all_ohlcv = []

        end_time = datetime.now()
        if start_time is None:
            raise ValueError("Start time is required")
        # Generate safe filename
        start_str = start_time.strftime('%Y%m%d_%H%M%S') if isinstance(start_time, datetime) else str(start_time)
        end_str = end_time.strftime('%Y%m%d_%H%M%S') if isinstance(end_time, datetime) else str(end_time)


        
        filename = f"spot_{symbol.replace('-', '_')}_ohlcv_{timeframe}_{start_str}_{end_str}.parquet"

        
        # Check if data already exists
        if Path(self.data_dir / filename).exists() and not export:
            df = pd.read_parquet(self.data_dir / filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.spot_ohlcv_data[symbol] = df
            return df
        else:
            logger.info(f"Collecting spot OHLCV for {symbol}...")

        ccxt_symbol = self._convert_symbol_to_ccxt(symbol, "spot")

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
            df['symbol'] = symbol
            df['market_type'] = 'spot'
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            if export:
                df.to_parquet(self.data_dir / filename, index=False)
                logger.info(f"  Saved spot OHLCV for {symbol}: {len(df):,} records to {filename}")
            self.spot_ohlcv_data[symbol] = df
            return df
        else:
            logger.error(f"  No spot OHLCV data collected for {symbol}")
            return None

    def collect_perpetual_mark_ohlcv(self, symbol: str, timeframe: str = '15m',
                                   start_time: datetime = None, export: bool = False):
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
            Whether to export data to Parquet file. Default is False.
        
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
        >>> mark_data = collector.collect_perpetual_mark_ohlcv("BTC-USDT", "1h", start_time, export=True)
        >>> print(mark_data.head())
        """
        max_records_per_request = 1000
        all_ohlcv = []

        end_time = datetime.now()
        if start_time is None:
            raise ValueError("Start time is required")

        # Generate safe filename
        start_str = start_time.strftime('%Y%m%d_%H%M%S') if isinstance(start_time, datetime) else str(start_time)
        end_str = end_time.strftime('%Y%m%d_%H%M%S') if isinstance(end_time, datetime) else str(end_time)
        filename = f"perpetual_{symbol.replace('-', '_')}_mark_{timeframe}_{start_str}_{end_str}.parquet"

        # Check if data already exists
        if Path(self.data_dir / filename).exists() and not export:
            df = pd.read_parquet(self.data_dir / filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.perpetual_mark_ohlcv_data[symbol] = df
            return df
        else:
            logger.info(f"Collecting perpetual mark price OHLCV for {symbol}...")

        ccxt_symbol = self._convert_symbol_to_ccxt(symbol, "future")

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
            df['symbol'] = symbol
            df['market_type'] = 'perpetual'
            df['price_type'] = 'mark'
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            if export:
                df.to_parquet(self.data_dir / filename, index=False)
                logger.info(f"  Saved perpetual mark OHLCV for {symbol}: {len(df):,} records to {filename}")
            self.perpetual_mark_ohlcv_data[symbol] = df
            return df
        else:
            logger.error(f"  No perpetual mark OHLCV data collected for {symbol}")
            return None

    def collect_perpetual_index_ohlcv(self, symbol: str, timeframe: str = '15m',
                                    start_time: datetime = None, export: bool = False):
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
            Whether to export data to Parquet file. Default is False.
        
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
        >>> index_data = collector.collect_perpetual_index_ohlcv("BTC-USDT", "1h", start_time, export=True)
        >>> print(index_data.head())
        """
        max_records_per_request = 1000
        all_ohlcv = []

        end_time = datetime.now()
        if start_time is None:
            raise ValueError("Start time is required")
        # Generate safe filename
        start_str = start_time.strftime('%Y%m%d_%H%M%S') if isinstance(start_time, datetime) else str(start_time)
        end_str = end_time.strftime('%Y%m%d_%H%M%S') if isinstance(end_time, datetime) else str(end_time)
        filename = f"perpetual_{symbol.replace('-', '_')}_index_{timeframe}_{start_str}_{end_str}.parquet"

        # Check if data already exists
        if Path(self.data_dir / filename).exists() and not export:
            df = pd.read_parquet(self.data_dir / filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.perpetual_index_ohlcv_data[symbol] = df
            return df
        else:
            logger.info(f"Collecting perpetual index price OHLCV for {symbol}...")

        ccxt_symbol = self._convert_symbol_to_ccxt(symbol, "future")

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
            df['symbol'] = symbol
            df['market_type'] = 'perpetual'
            df['price_type'] = 'index'
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            if export:
                df.to_parquet(self.data_dir / filename, index=False)
                logger.info(f"  Saved perpetual index OHLCV for {symbol}: {len(df):,} records to {filename}")
            self.perpetual_index_ohlcv_data[symbol] = df
            return df
        else:
            logger.error(f"  No perpetual index OHLCV data collected for {symbol}")
            return None

    def collect_funding_rates(self, symbol: str, start_time: datetime = None, export: bool = False):
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
            Whether to export data to Parquet file. Default is False.
        
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
        >>> funding_data = collector.collect_funding_rates("BTC-USDT", start_time, export=True)
        >>> print(funding_data.head())
        """
        end_time = datetime.now()
        if start_time is None:
            raise ValueError("Start time is required")

        # Generate safe filename
        start_str = start_time.strftime('%Y%m%d_%H%M%S') if isinstance(start_time, datetime) else str(start_time)
        end_str = end_time.strftime('%Y%m%d_%H%M%S') if isinstance(end_time, datetime) else str(end_time)
        filename = f"perpetual_{symbol.replace('-', '_')}_funding_rates_{start_str}_{end_str}.parquet"

        # Check if data already exists
        if Path(self.data_dir / filename).exists() and not export:
            df = pd.read_parquet(self.data_dir / filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.funding_rates_data[symbol] = df
            return df
        else:
            logger.info(f"Collecting funding rates for {symbol}...")

        ccxt_symbol = self._convert_symbol_to_ccxt(symbol, "future")
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
                        'symbol': symbol,
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
                    logger.info(f"  Saved funding rates for {symbol}: {len(df):,} records to {filename}")
                self.funding_rates_data[symbol] = df
                return df
            else:
                logger.error(f"  No funding rate data for {symbol}")
                return None
            
        except Exception as e:
            logger.error(f"Failed to collect funding rates for {symbol}: {e}")
            return None

    def collect_open_interest(self, symbol: str, timeframe: str = '15m',
                            start_time: datetime = None, export: bool = False):
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
            Whether to export data to Parquet file. Default is False.
        
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
        >>> oi_data = collector.collect_open_interest("BTC-USDT", "1h", start_time, export=True)
        >>> print(oi_data.head())
        """
        max_records_per_request = 1000
        all_open_interest = []

        end_time = datetime.now()
        if start_time is None:
            raise ValueError("Start time is required")

        # Generate safe filename
        start_str = start_time.strftime('%Y%m%d_%H%M%S') if isinstance(start_time, datetime) else str(start_time)
        end_str = end_time.strftime('%Y%m%d_%H%M%S') if isinstance(end_time, datetime) else str(end_time)
        filename = f"perpetual_{symbol.replace('-', '_')}_open_interest_{timeframe}_{start_str}_{end_str}.parquet"

        # Check if data already exists
        if Path(self.data_dir / filename).exists() and not export:
            df = pd.read_parquet(self.data_dir / filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.open_interest_data[symbol] = df
            return df
        else:
            logger.info(f"Collecting open interest for {symbol}...")

        ccxt_symbol = self._convert_symbol_to_ccxt(symbol, "future")

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
                    'symbol': symbol,
                    'market_type': 'perpetual'
                })
            
            df = pd.DataFrame(processed_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            if export:
                df.to_parquet(self.data_dir / filename, index=False)
                logger.info(f"  Saved open interest for {symbol}: {len(df):,} records to {filename}")
            self.open_interest_data[symbol] = df
            return df
        else:
            logger.error(f"  No open interest data for {symbol}")
            return None

    def collect_perpetual_trades(self, symbol: str, start_time: datetime = None, export: bool = False):
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
            Whether to export data to Parquet file. Default is False.
        
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
        >>> trades_data = collector.collect_perpetual_trades("BTC-USDT", start_time, export=True)
        >>> print(trades_data.head())
        """
        end_time = datetime.now()
        if start_time is None:
            raise ValueError("Start time is required")

        # Generate safe filename
        start_str = start_time.strftime('%Y%m%d_%H%M%S') if isinstance(start_time, datetime) else str(start_time)
        end_str = end_time.strftime('%Y%m%d_%H%M%S') if isinstance(end_time, datetime) else str(end_time)
        filename = f"perpetual_{symbol.replace('-', '_')}_trades_{start_str}_{end_str}.parquet"

        # Check if data already exists
        if Path(self.data_dir / filename).exists() and not export:
            df = pd.read_parquet(self.data_dir / filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.perpetual_trades_data[symbol] = df
            return df
        else:
            logger.info(f"Collecting perpetual trades for {symbol}...")

        ccxt_symbol = self._convert_symbol_to_ccxt(symbol, "future")
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
                    current_start_time = datetime.fromtimestamp(trades_time/1000)
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
                    'symbol': symbol,
                    'market_type': 'perpetual'
                })
            
            df = pd.DataFrame(processed_trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            if export:
                df.to_parquet(self.data_dir / filename, index=False)
                logger.info(f"  Saved perpetual trades for {symbol}: {len(df):,} records to {filename}")
            self.perpetual_trades_data[symbol] = df
            return df
        else:
            logger.error(f"  No perpetual trades data for {symbol}")
            return None



    async def collect_live_futures_ohlcv(self, symbol: str):
        symbol = self._convert_symbol_to_ccxt(symbol, "future")
        try:
            data = await self.live_futures_exchange.watch_ohlcv(symbol)
            rows = data if (data and isinstance(data[0], (list, tuple))) else [data]
            df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Failed to collect live futures OHLCV data for {symbol}: {e}")
            return None

    async def collect_live_spot_ohlcv(self, symbol: str):
        symbol = self._convert_symbol_to_ccxt(symbol, "spot")
        try:
            data = await self.live_spot_exchange.watch_ohlcv(symbol)
            rows = data if (data and isinstance(data[0], (list, tuple))) else [data]
            df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Failed to collect live spot OHLCV data for {symbol}: {e}")
            return None

    # Helper methods with documentation
    def _convert_symbol_to_ccxt(self, symbol: str, market_type: str = "spot"):
        """
        Convert trading symbol format for CCXT API calls.
        
        Parameters
        ----------
        symbol : str
            Trading symbol in standard format (e.g., "BTC-USDT", "ETH-USDT-PERP")
        market_type : str, optional
            Market type: "spot" or "future". Default is "spot".
        
        Returns
        -------
        str or None
            Converted symbol format for CCXT API calls.
            For spot: "BTC-USDT" -> "BTC/USDT"
            For futures: "BTC-USDT-PERP" -> "BTC/USDT:USDT"
            Returns None if conversion fails.
        """
        try:
            if market_type == "spot":
                # For spot: BTC-USDT -> BTC/USDT
                if '-' in symbol:
                    return symbol.replace('-', '/')
                return symbol
            else:
                # For futures: BTC-USDT-PERP -> BTC/USDT:USDT
                base = symbol.split('-')[0]
                return f"{base}/USDT:USDT"
        except:
            return None

    def _get_number_of_periods(self, timeframe: str, start_time: datetime, end_time: datetime):
        """
        Calculate the number of periods between two datetime objects for a given timeframe.
        
        Parameters
        ----------
        timeframe : str
            Timeframe string (e.g., '1m', '5m', '15m', '1h', '1d')
        start_time : datetime
            Start datetime
        end_time : datetime
            End datetime
        
        Returns
        -------
        int
            Number of periods between start_time and end_time
        """
        minutes = self._get_timeframe_to_minutes(timeframe)
        total_minutes = (end_time - start_time).total_seconds() / 60
        total_periods = int(total_minutes // minutes)
        return total_periods

    def _get_timeframe_to_minutes(self, timeframe: str):
        """
        Convert timeframe string to minutes.
        
        Parameters
        ----------
        timeframe : str
            Timeframe string (e.g., '1m', '5m', '15m', '1h', '1d')
        
        Returns
        -------
        int
            Number of minutes for the timeframe. Defaults to 15 if timeframe not recognized.
        """
        periods_map = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360,
            '12h': 720, '1d': 1440,
        }
        return periods_map.get(timeframe, 15)

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
        # Calculate periods per day based on timeframe
        total_periods = self._get_number_of_periods(timeframe, start_time, end_time)
        num_batches = (total_periods + limit - 1) // limit
        minutes = self._get_timeframe_to_minutes(timeframe)
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
                        logger.info(f"  Data collection batch {batch+1}/{num_batches}: {len(data)} records")
                else:
                    break
                
                time.sleep(0.1)
                
            except Exception as e:
                if logger:
                    logger.error(f"  Error in data collection batch {batch+1}: {e}")
                time.sleep(1)
                continue
        
        return all_data