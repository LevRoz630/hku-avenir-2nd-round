#!/usr/bin/env python3
"""
Historical data layer:
- HistoricalDataManager: loads CSVs into memory and serves time-filtered slices
- HistoricalDataCollector: optionally downloads CSVs from CCXT when missing
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
from ccxt import binance as binance_sync
import os
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HistoricalDataManager:
    """Load and serve historical data required by the backtester/OMS."""
    
    def __init__(self, data_dir: str = "historical_data"):
        self.data_dir = Path(data_dir)
        
        # Separate data storage for different market types and data types
        self.spot_ohlcv_data = {}
        self.spot_trades_data = {}
        self.perpetual_mark_ohlcv_data = {}
        self.perpetual_index_ohlcv_data = {}
        self.funding_data = {}
        self.open_interest_data = {}
        self.perpetual_trades_data = {}
        
    def load_historical_data(self, symbols: List[str], data_types: List[str] = None):
        """Load all historical data for given symbols"""
        if data_types is None:
            data_types = ['spot_ohlcv', 'spot_trades', 'perpetual_mark_ohlcv', 'perpetual_index_ohlcv', 'funding_rates', 'open_interest', 'perpetual_trades']
            
        logger.info(f"Loading historical data for {len(symbols)} symbols...")
        
        for symbol in symbols:
            # Load spot data
            if 'spot_ohlcv' in data_types:
                self.spot_ohlcv_data[symbol] = self._load_spot_ohlcv_data(symbol)
            # if 'spot_trades' in data_types:
            #     self.spot_trades_data[symbol] = self._load_spot_trades_data(symbol)
            
            # Load perpetual data
            if 'perpetual_mark_ohlcv' in data_types:
                self.perpetual_mark_ohlcv_data[symbol] = self._load_perpetual_mark_ohlcv_data(symbol)
            if 'perpetual_index_ohlcv' in data_types:
                self.perpetual_index_ohlcv_data[symbol] = self._load_perpetual_index_ohlcv_data(symbol)
            if 'funding_rates' in data_types:
                self.funding_data[symbol] = self._load_funding_data(symbol)
            if 'open_interest' in data_types:
                self.open_interest_data[symbol] = self._load_open_interest_data(symbol)
            # if 'perpetual_trades' in data_types:
            #     self.perpetual_trades_data[symbol] = self._load_perpetual_trades_data(symbol)
                
        logger.info("Comprehensive historical data loading completed")
    
    def _load_spot_ohlcv_data(self, symbol: str) -> pd.DataFrame:
        """Load spot OHLCV data for a symbol"""
        from glob import glob
        for timeframe in ['1m', '5m', '15m', '1h', '1d']:
            # Accept any days suffix (e.g., 2d, 30d, 90d)
            pattern = str(self.data_dir / f"spot_{symbol}_{timeframe}_*d.csv")
            matches = glob(pattern)
            if matches:
                filepath = Path(sorted(matches)[-1])
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        
        logger.warning(f"No spot OHLCV data found for {symbol}")
        return pd.DataFrame()
    
    def _load_spot_trades_data(self, symbol: str) -> pd.DataFrame:
        """Load spot trades data for a symbol"""
        from glob import glob
        pattern = str(self.data_dir / f"spot_{symbol}_trades_*d.csv")
        matches = glob(pattern)
        if matches:
            filepath = Path(sorted(matches)[-1])
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        logger.warning(f"No spot trades data found for {symbol}")
        return pd.DataFrame()
    
    def _load_perpetual_mark_ohlcv_data(self, symbol: str) -> pd.DataFrame:
        """Load perpetual mark price OHLCV data for a symbol"""
        from glob import glob
        for timeframe in ['1m', '5m', '15m', '1h', '1d']:
            pattern = str(self.data_dir / f"perpetual_{symbol}_mark_{timeframe}_*d.csv")
            matches = glob(pattern)
            if matches:
                filepath = Path(sorted(matches)[-1])
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        
        logger.warning(f"No perpetual mark OHLCV data found for {symbol}")
        return pd.DataFrame()
    
    def _load_perpetual_index_ohlcv_data(self, symbol: str) -> pd.DataFrame:
        """Load perpetual index price OHLCV data for a symbol"""
        from glob import glob
        for timeframe in ['1m', '5m', '15m', '1h', '1d']:
            pattern = str(self.data_dir / f"perpetual_{symbol}_index_{timeframe}_*d.csv")
            matches = glob(pattern)
            if matches:
                filepath = Path(sorted(matches)[-1])
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        
        logger.warning(f"No perpetual index OHLCV data found for {symbol}")
        return pd.DataFrame()
    
    def _load_funding_data(self, symbol: str) -> pd.DataFrame:
        """Load funding rate data for a symbol"""
        from glob import glob
        pattern = str(self.data_dir / f"perpetual_{symbol}_funding_rates_*d.csv")
        matches = glob(pattern)
        if matches:
            filepath = Path(sorted(matches)[-1])
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        logger.warning(f"No funding rate data found for {symbol}")
        return pd.DataFrame()
    
    def _load_open_interest_data(self, symbol: str) -> pd.DataFrame:
        """Load open interest data for a symbol"""
        from glob import glob
        for timeframe in ['5m', '15m', '1h']:
            pattern = str(self.data_dir / f"perpetual_{symbol}_open_interest_*d_{timeframe}.csv")
            matches = glob(pattern)
            if matches:
                filepath = Path(sorted(matches)[-1])
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        
        logger.warning(f"No open interest data found for {symbol}")
        return pd.DataFrame()
    
    def _load_perpetual_trades_data(self, symbol: str) -> pd.DataFrame:
        """Load perpetual trades data for a symbol"""
        from glob import glob
        pattern = str(self.data_dir / f"perpetual_{symbol}_trades_*d.csv")
        matches = glob(pattern)
        if matches:
            filepath = Path(sorted(matches)[-1])
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        logger.warning(f"No perpetual trades data found for {symbol}")
        return pd.DataFrame()
    
    def get_data_at_time(self, symbol: str, timestamp: datetime, data_type: str = 'spot_ohlcv') -> pd.DataFrame:
        """Get data up to a specific timestamp"""
        if data_type == 'spot_ohlcv':
            data = self.spot_ohlcv_data.get(symbol, pd.DataFrame())
        elif data_type == 'spot_trades':
            data = self.spot_trades_data.get(symbol, pd.DataFrame())
        elif data_type == 'perpetual_mark_ohlcv':
            data = self.perpetual_mark_ohlcv_data.get(symbol, pd.DataFrame())
        elif data_type == 'perpetual_index_ohlcv':
            data = self.perpetual_index_ohlcv_data.get(symbol, pd.DataFrame())
        elif data_type == 'funding_rates':
            data = self.funding_data.get(symbol, pd.DataFrame())
        elif data_type == 'open_interest':
            data = self.open_interest_data.get(symbol, pd.DataFrame())
        elif data_type == 'perpetual_trades':
            data = self.perpetual_trades_data.get(symbol, pd.DataFrame())
        else:
            return pd.DataFrame()
        
        if data.empty:
            return data
            
        # Filter data up to timestamp
        return data[data['timestamp'] <= timestamp].copy()
    
    def get_comprehensive_data_at_time(self, symbol: str, timestamp: datetime) -> Dict[str, pd.DataFrame]:
        """Get all available data types for a symbol at a specific timestamp"""
        return {
            'spot_ohlcv': self.get_data_at_time(symbol, timestamp, 'spot_ohlcv'),
            'spot_trades': self.get_data_at_time(symbol, timestamp, 'spot_trades'),
            'perpetual_mark_ohlcv': self.get_data_at_time(symbol, timestamp, 'perpetual_mark_ohlcv'),
            'perpetual_index_ohlcv': self.get_data_at_time(symbol, timestamp, 'perpetual_index_ohlcv'),
            'funding_rates': self.get_data_at_time(symbol, timestamp, 'funding_rates'),
            'open_interest': self.get_data_at_time(symbol, timestamp, 'open_interest'),
            'perpetual_trades': self.get_data_at_time(symbol, timestamp, 'perpetual_trades')
        }



class HistoricalDataCollector:

    def __init__(self, data_dir="historical_data", days=None, symbols=None):
        """Initialize historical data collector for both spot and perpetual data"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.days = days
        self.symbols = symbols

        # Initialize separate exchanges for spot and futures
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

        logger.info(f"Historical Data Collector initialized for {days} days with {len(symbols)} symbols")
    
    def _convert_symbol_to_ccxt(self, symbol, market_type="spot"):
        """Convert symbol format for CCXT API calls"""
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
    
    def collect_comprehensive_data(self, timeframe='15m'):
        """Collect comprehensive data for both spot and perpetual markets"""
        logger.info(f"Collecting comprehensive data for {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            try:
                logger.info(f"Processing {symbol}...")
                
                # Collect spot data
                self._collect_spot_data(symbol, timeframe)
                
                # Collect perpetual futures data (mark price, index price, funding rates, open interest, trades)
                self._collect_perpetual_data(symbol, timeframe)
                
                # Rate limiting between symbols
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to collect data for {symbol}: {e}")
    
    def _collect_spot_data(self, symbol, timeframe='15m'):
        """Collect spot market data (OHLCV and trades)"""
        logger.info(f"Collecting spot data for {symbol}...")
        
        # Convert symbol format for spot
        ccxt_symbol = self._convert_symbol_to_ccxt(symbol, "spot")
        if not ccxt_symbol:
            logger.error(f"Could not convert {symbol} for spot trading")
            return
        
        try:
            # Collect spot OHLCV
            self._collect_spot_ohlcv(symbol, ccxt_symbol, timeframe)
            
            # Collect spot trades
            # self._collect_spot_trades(symbol, ccxt_symbol)
            
        except Exception as e:
            logger.error(f"Failed to collect spot data for {symbol}: {e}")
    
    def _collect_perpetual_data(self, symbol, timeframe='15m'):
        """Collect perpetual futures data (mark price, index price, funding rates, open interest, trades)"""
        logger.info(f"Collecting perpetual data for {symbol}...")
        
        # Convert symbol format for futures
        ccxt_symbol = self._convert_symbol_to_ccxt(symbol, "future")
        if not ccxt_symbol:
            logger.error(f"Could not convert {symbol} for futures trading")
            return
        
        try:
            # Collect perpetual mark price OHLCV
            self._collect_perpetual_mark_ohlcv(symbol, ccxt_symbol, timeframe)
            
            # Collect perpetual index price OHLCV
            self._collect_perpetual_index_ohlcv(symbol, ccxt_symbol, timeframe)
            
            # Collect funding rates
            self._collect_funding_rates(symbol, ccxt_symbol)
            
            # Collect open interest
            self._collect_open_interest(symbol, ccxt_symbol, timeframe)
            
            # Collect perpetual trades
            # self._collect_perpetual_trades(symbol, ccxt_symbol)
            
        except Exception as e:
            logger.error(f"Failed to collect perpetual data for {symbol}: {e}")
    
    def _collect_spot_ohlcv(self, symbol, ccxt_symbol, timeframe='15m'):
        """Collect spot OHLCV data"""
        logger.info(f"Collecting spot OHLCV for {symbol}...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.days)
        
        # Calculate periods per day based on timeframe
        periods_per_day = self._get_periods_per_day(timeframe)
        total_periods = self.days * periods_per_day
        max_records_per_request = 1000
        
        all_ohlcv = []
        current_end_time = end_time
        
        num_batches = (total_periods + max_records_per_request - 1) // max_records_per_request
        
        for batch in range(num_batches):
            try:
                batch_days = min(self.days, max_records_per_request / periods_per_day)
                batch_start_time = current_end_time - timedelta(days=batch_days)
                since = int(batch_start_time.timestamp() * 1000)
                
                ohlcv = self.spot_exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=since, limit=max_records_per_request)
                
                if ohlcv:
                    all_ohlcv.extend(ohlcv)
                    current_end_time = batch_start_time
                    logger.info(f"  Spot OHLCV batch {batch+1}/{num_batches}: {len(ohlcv)} records")
                else:
                    break
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"  Error in spot OHLCV batch {batch+1}: {e}")
                time.sleep(1)
                continue
        
        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            df['market_type'] = 'spot'
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            filename = f"spot_{symbol}_{timeframe}_{self.days}d.csv"
            df.to_csv(self.data_dir / filename, index=False)
            logger.info(f"  Saved spot OHLCV for {symbol}: {len(df):,} records to {filename}")
        else:
            logger.warning(f"  No spot OHLCV data collected for {symbol}")
    
    def _collect_spot_trades(self, symbol, ccxt_symbol):
        """Collect spot trades data"""
        logger.info(f"Collecting spot trades for {symbol}...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.days)
        
        all_trades = []
        current_start_time = start_time
        
        while current_start_time < end_time:
            try:
                since = int(current_start_time.timestamp() * 1000)
                trades = self.spot_exchange.fetch_trades(ccxt_symbol, since=since, limit=1000)
                
                if trades:
                    all_trades.extend(trades)
                    logger.info(f"  Spot trades batch: {len(trades)} trades")
                    trades_time = trades[-1].get('timestamp')
                    current_start_time = datetime.fromtimestamp(trades_time/1000)
                else:
                    break
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"  Error in spot trades batch: {e}")
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
                    'market_type': 'spot'
                })
            
            df = pd.DataFrame(processed_trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            filename = f"spot_{symbol}_trades_{self.days}d.csv"
            df.to_csv(self.data_dir / filename, index=False)
            logger.info(f"  Saved spot trades for {symbol}: {len(df):,} records to {filename}")
        else:
            logger.warning(f"  No spot trades data for {symbol}")
    
    def _collect_perpetual_mark_ohlcv(self, symbol, ccxt_symbol, timeframe='15m'):
        """Collect perpetual mark price OHLCV data"""
        logger.info(f"Collecting perpetual mark price OHLCV for {symbol}...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.days)
        
        periods_per_day = self._get_periods_per_day(timeframe)
        total_periods = self.days * periods_per_day
        max_records_per_request = 1000
        
        all_ohlcv = []
        current_end_time = end_time
        
        num_batches = (total_periods + max_records_per_request - 1) // max_records_per_request
        
        for batch in range(num_batches):
            try:
                batch_days = min(self.days, max_records_per_request / periods_per_day)
                batch_start_time = current_end_time - timedelta(days=batch_days)
                since = int(batch_start_time.timestamp() * 1000)
                
                ohlcv = self.futures_exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=since, limit=max_records_per_request, params={'price': 'mark'})
                
                if ohlcv:
                    all_ohlcv.extend(ohlcv)
                    current_end_time = batch_start_time
                    logger.info(f"  Perpetual mark OHLCV batch {batch+1}/{num_batches}: {len(ohlcv)} records")
                else:
                    break
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"  Error in perpetual mark OHLCV batch {batch+1}: {e}")
                time.sleep(1)
                continue
        
        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            df['market_type'] = 'perpetual'
            df['price_type'] = 'mark'
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            filename = f"perpetual_{symbol}_mark_{timeframe}_{self.days}d.csv"
            df.to_csv(self.data_dir / filename, index=False)
            logger.info(f"  Saved perpetual mark OHLCV for {symbol}: {len(df):,} records to {filename}")
        else:
            logger.warning(f"  No perpetual mark OHLCV data collected for {symbol}")
    
    def _collect_perpetual_index_ohlcv(self, symbol, ccxt_symbol, timeframe='15m'):
        """Collect perpetual index price OHLCV data"""
        logger.info(f"Collecting perpetual index price OHLCV for {symbol}...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.days)
        
        periods_per_day = self._get_periods_per_day(timeframe)
        total_periods = self.days * periods_per_day
        max_records_per_request = 1000
        
        all_ohlcv = []
        current_end_time = end_time
        
        num_batches = (total_periods + max_records_per_request - 1) // max_records_per_request
        
        for batch in range(num_batches):
            try:
                batch_days = min(self.days, max_records_per_request / periods_per_day)
                batch_start_time = current_end_time - timedelta(days=batch_days)
                since = int(batch_start_time.timestamp() * 1000)
                
                ohlcv = self.futures_exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=since, limit=max_records_per_request, params={'price': 'index'})
                
                if ohlcv:
                    all_ohlcv.extend(ohlcv)
                    current_end_time = batch_start_time
                    logger.info(f"  Perpetual index OHLCV batch {batch+1}/{num_batches}: {len(ohlcv)} records")
                else:
                    break
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"  Error in perpetual index OHLCV batch {batch+1}: {e}")
                time.sleep(1)
                continue
        
        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            df['market_type'] = 'perpetual'
            df['price_type'] = 'index'
            df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
            
            filename = f"perpetual_{symbol}_index_{timeframe}_{self.days}d.csv"
            df.to_csv(self.data_dir / filename, index=False)
            logger.info(f"  Saved perpetual index OHLCV for {symbol}: {len(df):,} records to {filename}")
        else:
            logger.warning(f"  No perpetual index OHLCV data collected for {symbol}")
    
    def _collect_funding_rates(self, symbol, ccxt_symbol):
        """Collect funding rates data"""
        logger.info(f"Collecting funding rates for {symbol}...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.days)
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
                    processed_data.append({
                        'timestamp': rate.get('timestamp'),
                        'funding_rate': rate.get('fundingRate'),
                        'funding_time': rate.get('fundingTime'),
                        'mark_price': rate.get('markPrice'),
                        'index_price': rate.get('indexPrice'),
                        'symbol': symbol,
                        'market_type': 'perpetual'
                    })
                
                df = pd.DataFrame(processed_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
                
                filename = f"perpetual_{symbol}_funding_rates_{self.days}d.csv"
                df.to_csv(self.data_dir / filename, index=False)
                logger.info(f"  Saved funding rates for {symbol}: {len(df):,} records to {filename}")
            else:
                logger.warning(f"  No funding rate data for {symbol}")
            
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Failed to collect funding rates for {symbol}: {e}")
    
    def _collect_open_interest(self, symbol, ccxt_symbol, timeframe='5m'):
        """Collect open interest data"""
        logger.info(f"Collecting open interest for {symbol}...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.days)
        
        all_open_interest = []
        batch_size = 500
        
        while start_time < end_time:
            try:
                open_interest = self.futures_exchange.fetch_open_interest_history(
                    ccxt_symbol,
                    since=int(start_time.timestamp() * 1000),
                    limit=batch_size,
                    timeframe=timeframe
                )
                
                if not open_interest:
                    logger.info(f"  No more open interest data available")
                    break
                
                all_open_interest.extend(open_interest)
                
                # Timeframe to timedelta mapping
                timeframe_deltas = {
                    '5m': timedelta(minutes=5),
                    '15m': timedelta(minutes=15),
                    '30m': timedelta(minutes=30),
                    '1h': timedelta(hours=1),
                    '2h': timedelta(hours=2),
                    '4h': timedelta(hours=4),
                    '6h': timedelta(hours=6),
                    '12h': timedelta(hours=12),
                    '1d': timedelta(days=1)
                }

                if timeframe in timeframe_deltas:
                    start_time = start_time + (timeframe_deltas[timeframe] * batch_size)
                else:
                    raise ValueError(f"Unsupported timeframe: {timeframe}")

                logger.info(f"  Fetched {len(open_interest)} open interest records (total: {len(all_open_interest)})")
                
            except Exception as e:
                logger.error(f"Error in open interest batch: {e}")
                break
        
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
            
            filename = f"perpetual_{symbol}_open_interest_{self.days}d_{timeframe}.csv"
            df.to_csv(self.data_dir / filename, index=False)
            logger.info(f"  Saved open interest for {symbol}: {len(df):,} records to {filename}")
        else:
            logger.warning(f"  No open interest data for {symbol}")
        
        time.sleep(2)
    
    def _collect_perpetual_trades(self, symbol, ccxt_symbol):
        """Collect perpetual trades data"""
        logger.info(f"Collecting perpetual trades for {symbol}...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.days)
        
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
            
            filename = f"perpetual_{symbol}_trades_{self.days}d.csv"
            df.to_csv(self.data_dir / filename, index=False)
            logger.info(f"  Saved perpetual trades for {symbol}: {len(df):,} records to {filename}")
        else:
            logger.warning(f"  No perpetual trades data for {symbol}")
    
    def _get_periods_per_day(self, timeframe):
        """Get number of periods per day for a given timeframe"""
        periods_per_day_map = {
            '1m': 1440,
            '5m': 288,
            '15m': 96,
            '30m': 48,
            '1h': 24,
            '2h': 12,
            '4h': 6,
            '6h': 4,
            '12h': 2,
            '1d': 1
        }
        return periods_per_day_map.get(timeframe, 96)  # Default to 15m if not found
    
    def collect_historical_ohlcv(self, timeframe='15m'):
        """Collect historical OHLCV data for multiple days"""

        logger.info(f"Collecting historical OHLCV data ({timeframe}) for {len(self.symbols)} symbols...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.days)
        
        # Calculate periods per day based on timeframe
        if timeframe == '1h':
            periods_per_day = 24
        elif timeframe == '15m':
            periods_per_day = 96
        elif timeframe == '1m':
            periods_per_day = 1440
        else:
            periods_per_day = 96
        
        total_periods = self.days * periods_per_day
        max_records_per_request = 1000  # Binance limit
        num_batches = (total_periods + max_records_per_request - 1) // max_records_per_request
        
        logger.info(f"Total periods needed: {total_periods:,}")
        logger.info(f"Number of batches: {num_batches}")
        
        for symbol in self.symbols:
            try:
                base = symbol.split('-')[0]
                ccxt_symbol = f"{base}/USDT"
                
                logger.info(f"Collecting {symbol}...")
                all_ohlcv = []
                current_end_time = end_time
                
                # Collect data in batches going backwards in time
                for batch in range(num_batches):
                    try:
                        # Calculate time range for this batch
                        batch_days = min(self.days, max_records_per_request / periods_per_day)
                        batch_start_time = current_end_time - timedelta(days=batch_days)
                        since = int(batch_start_time.timestamp() * 1000)
                        
                        ohlcv = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=since, limit=max_records_per_request, params={'price': 'mark'})
                        
                        if ohlcv:
                            all_ohlcv.extend(ohlcv)
                            current_end_time = batch_start_time
                            logger.info(f"  Batch {batch+1}/{num_batches}: {len(ohlcv)} records")
                        else:
                            logger.warning(f"  No data in batch {batch+1}")
                            break
                        
                        # Rate limiting
                        time.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"  Error in batch {batch+1}: {e}")
                        time.sleep(1)
                        continue
                
                if all_ohlcv:
                    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['symbol'] = symbol
                    df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
                    
                    # Save individual symbol data
                    filename = f"{self.type}_{symbol}_{timeframe}_{self.days}d.csv"
                    df.to_csv(self.data_dir / filename, index=False)
                    logger.info(f"  Saved {symbol}: {len(df):,} records to {filename}")
                else:
                    logger.warning(f"  No data collected for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to collect {symbol}: {e}")

    def collect_historical_ohlcv_index(self, timeframe='15m'):
        """Collect historical OHLCV index data for multiple days"""

        logger.info(f"Collecting historical OHLCV index data ({timeframe}) for {len(self.symbols)} symbols...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.days)
        
        # Calculate periods per day based on timeframe
        if timeframe == '1h':
            periods_per_day = 24
        elif timeframe == '15m':
            periods_per_day = 96
        elif timeframe == '1m':
            periods_per_day = 1440
        else:
            periods_per_day = 96
        
        total_periods = self.days * periods_per_day
        max_records_per_request = 1000  # Binance limit
        num_batches = (total_periods + max_records_per_request - 1) // max_records_per_request
        
        logger.info(f"Total periods needed: {total_periods:,}")
        logger.info(f"Number of batches: {num_batches}")
        
        for symbol in self.symbols:
            try:
                base = symbol.split('-')[0]
                ccxt_symbol = f"{base}/USDT"
                
                logger.info(f"Collecting {symbol}...")
                all_ohlcv = []
                current_end_time = end_time
                
                # Collect data in batches going backwards in time
                for batch in range(num_batches):
                    try:
                        # Calculate time range for this batch
                        batch_days = min(self.days, max_records_per_request / periods_per_day)
                        batch_start_time = current_end_time - timedelta(days=batch_days)
                        since = int(batch_start_time.timestamp() * 1000)
                        
                        ohlcv = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=since, limit=max_records_per_request, params={'price': 'index'})
                        
                        if ohlcv:
                            all_ohlcv.extend(ohlcv)
                            current_end_time = batch_start_time
                            logger.info(f"  Batch {batch+1}/{num_batches}: {len(ohlcv)} records")
                        else:
                            logger.warning(f"  No data in batch {batch+1}")
                            break
                        
                        # Rate limiting
                        time.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"  Error in batch {batch+1}: {e}")
                        time.sleep(1)
                        continue
                
                if all_ohlcv:
                    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['symbol'] = symbol
                    df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
                    
                    # Save individual symbol data
                    filename = f"{self.type}_{symbol}_{timeframe}_{self.days}d.csv"
                    df.to_csv(self.data_dir / filename, index=False)
                    logger.info(f"  Saved {symbol}: {len(df):,} records to {filename}")
                else:
                    logger.warning(f"  No data collected for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to collect {symbol}: {e}")

    def collect_historical_funding_rates(self):
        if self.type == "spot":
            raise ValueError("spot does not have funding rates dumbass")
        """Collect historical funding rates for multiple days"""
        logger.info(f"Collecting historical funding rates for {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            try:
                base = symbol.split('-')[0]
                ccxt_symbol = f"{base}/USDT"
                
                logger.info(f"Collecting funding rates for {symbol}...")
                
                # Calculate time range
                end_time = datetime.now()
                start_time = end_time - timedelta(days=self.days)
                
                # Fetch funding rate history
                funding_rates = self.exchange.fetch_funding_rate_history(
                    ccxt_symbol, 
                    since=int(start_time.timestamp() * 1000),
                    limit=1000,
                )
                
                if funding_rates:
                    # Process funding rates data properly
                    processed_data = []
                    for rate in funding_rates:
                        processed_data.append({
                            'timestamp': rate.get('timestamp'),
                            'funding_rate': rate.get('fundingRate'),
                            'funding_time': rate.get('fundingTime'),
                            'mark_price': rate.get('markPrice'),
                            'index_price': rate.get('indexPrice'),
                            'symbol': symbol
                        })
                    
                    df = pd.DataFrame(processed_data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
                    
                    # Save funding rates
                    filename = f"{self.type}_{symbol}_funding_rates_{self.days}d.csv"
                    df.to_csv(self.data_dir / filename, index=False)
                    logger.info(f"  Saved {symbol}: {len(df):,} funding rate records to {filename}")
                else:
                    logger.warning(f"  No funding rate data for {symbol}")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Failed to collect funding rates for {symbol}: {e}")

    def collect_historical_open_interest(self, timeframe='5m'):
        if self.type == "spot":
            raise ValueError("spot does not have open interest dumbass")
        """Collect historical open interest data with proper API rate limiting"""
        logger.info(f"Collecting historical open interest for {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            try:
                base = symbol.split('-')[0]
                ccxt_symbol = f"{base}/USDT"
                
                logger.info(f"Collecting open interest for {symbol}...")
                all_open_interest = []
                batch_size = 500
                # Calculate time range - use shorter period to respect API limits
                end_time = datetime.now()
                start_time = end_time - timedelta(days=self.days)
                while start_time < end_time:


                    open_interest = self.exchange.fetch_open_interest_history(
                        ccxt_symbol,
                        since=int(start_time.timestamp() * 1000),
                        limit=batch_size,
                        timeframe=timeframe# Reduced limit to avoid API errors
                        )
                        
                    if not open_interest:
                        logger.info(f"  No more data available")
                        break
                    
                    all_open_interest.extend(open_interest)
                    
                    # Or if you want to use the timeframe parameter
                    # Timeframe to timedelta mapping
                    timeframe_deltas = {
                        '5m': timedelta(minutes=5),
                        '15m': timedelta(minutes=15),
                        '30m': timedelta(minutes=30),
                        '1h': timedelta(hours=1),
                        '2h': timedelta(hours=2),
                        '4h': timedelta(hours=4),
                        '6h': timedelta(hours=6),
                        '12h': timedelta(hours=12),
                        '1d': timedelta(days=1)
                    }

                    # Apply the timeframe shift
                    if timeframe in timeframe_deltas:
                        start_time = start_time + (timeframe_deltas[timeframe] * batch_size)
                    else:
                        raise ValueError(f"Unsupported timeframe got fix it : {timeframe}")

                    logger.info(f"  Fetched {len(open_interest)} records (total: {len(all_open_interest)})")
                    
                if all_open_interest:
                    # Process open interest data properly
                    processed_data = []
                    for oi in all_open_interest:
                        processed_data.append({
                            'timestamp': oi.get('timestamp'),
                            'open_interest': oi.get('openInterestAmount'),
                            'open_interest_value': oi.get('openInterestValue'),
                            'contract_type': oi.get('contractType', 'perpetual'),
                            'symbol': symbol
                        })
                    
                    df = pd.DataFrame(processed_data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
                    
                    # Save open interest
                    filename = f"{self.type}_{symbol}_open_interest_{self.days}d_{timeframe}.csv"
                    df.to_csv(self.data_dir / filename, index=False)
                    logger.info(f"  Saved {symbol}: {len(df):,} open interest records to {filename}")
                else:
                    logger.warning(f"  No open interest data for {symbol}")
                
                # Longer delay between symbols
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to collect open interest for {symbol}: {e}")

    def collect_historical_trades(self):
        """Collect historical trades data (limited to recent period due to API limits)"""
        logger.info(f"Collecting historical trades for {len(self.symbols)} symbols (last {self.days} days)...")
        
        for symbol in self.symbols:
            try:
                base = symbol.split('-')[0]
                ccxt_symbol = f"{base}/USDT"
                
                logger.info(f"Collecting trades for {symbol}...")
                
                # Calculate time range (limited to recent period)
                end_time = datetime.now()
                start_time = end_time - timedelta(days=self.days)
                
                all_trades = []
                current_end_time = end_time
                
                # Collect trades in batches
                while start_time < end_time:
                    try:
                        since = int(start_time.timestamp() * 1000)
                        trades = self.exchange.fetch_trades(ccxt_symbol, since=since, limit=1000)
                        
                        if trades:
                            all_trades.extend(trades)
                            logger.info(f"  Batch {start_time}: {len(trades)} trades")
                            trades_time = trades[-1].get('timestamp')
                            start_time = datetime.fromtimestamp(trades_time/1000)
                        else:
                            break
                        
                        time.sleep(0.5)  # Rate limiting
                        
                    except Exception as e:
                        logger.error(f"  Error in trades batch {start_time}: {e}")
                        break
                
                if all_trades:
                    # Process trades data properly
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
                            'symbol': symbol
                        })
                    
                    df = pd.DataFrame(processed_trades)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
                    
                    # Save trades
                    filename = f"{self.type}_{symbol}_trades_{self.days}d.csv"
                    df.to_csv(self.data_dir / filename, index=False)
                    logger.info(f"  Saved {symbol}: {len(df):,} trade records to {filename}")
                else:
                    logger.warning(f"  No trades data for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to collect trades for {symbol}: {e}")