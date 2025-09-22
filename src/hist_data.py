#!/usr/bin/env python3
"""
Historical data layer:
- HistoricalDataCollector: optionally downloads parquet from CCXT when missing
"""

import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
from ccxt import binance as binance_sync

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HistoricalDataCollector:

    def __init__(self, data_dir="../hku-data/test_data"):
        """Initialize historical data collector for both spot and perpetual data"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.spot_ohlcv_data = {}
        self.perpetual_mark_ohlcv_data = {}
        self.perpetual_index_ohlcv_data = {}
        self.spot_trades_data = {}
        self.perpetual_trades_data = {}
        self.funding_rates_data = {}
        self.open_interest_data = {}
        
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

    
    def convert_symbol_to_ccxt(self, symbol, market_type="spot"):
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
    
    
    def collect_spot_ohlcv(self, symbol, timeframe='15m', days=None):
        """Collect spot OHLCV data"""
        filename = f"spot_{symbol}_ohlcv_{timeframe}_{days}d.parquet"

        if Path(self.data_dir / filename).exists():
            df = pd.read_parquet(self.data_dir / filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.spot_ohlcv_data[symbol] = df
            return df
        else:
            logger.info(f"Collecting spot OHLCV for {symbol}...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        ccxt_symbol = self.convert_symbol_to_ccxt(symbol, "spot")
        # Calculate periods per day based on timeframe
        periods_per_day = self.get_periods_per_day(timeframe)
        total_periods = days * periods_per_day
        max_records_per_request = 1000
        
        all_ohlcv = []
        current_end_time = end_time
        
        num_batches = (total_periods + max_records_per_request - 1) // max_records_per_request
        
        for batch in range(num_batches):
            try:
                batch_days = min(days, max_records_per_request / periods_per_day)
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
            
            df.to_parquet(self.data_dir / filename, index=False)
            logger.info(f"  Saved spot OHLCV for {symbol}: {len(df):,} records to {filename}")
            self.spot_ohlcv_data[symbol] = df
            return df
        else:
            logger.error(f"  No spot OHLCV data collected for {symbol}")
    
    def collect_spot_trades(self, symbol, days=None):
        """Collect spot trades data"""
        filename = f"spot_{symbol}_trades_{days}d.parquet"

        if Path(self.data_dir / filename).exists():
            df = pd.read_parquet(self.data_dir / filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.spot_trades_data[symbol] = df
            return df
        else:
            logger.info(f"Collecting spot trades for {symbol}...")

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        ccxt_symbol = self.convert_symbol_to_ccxt(symbol, "spot")
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
            
            
            df.to_parquet(self.data_dir / filename, index=False)
            logger.info(f"  Saved spot trades for {symbol}: {len(df):,} records to {filename}")
            self.spot_trades_data[symbol] = df
            return df
        else:
            logger.error(f"  No spot trades data for {symbol}")
    
    def collect_perpetual_mark_ohlcv(self, symbol, timeframe='15m', days=None):
        """Collect perpetual mark price OHLCV data"""
        print(f"DEBUG collect_perp_mark inputs symbol={symbol} timeframe={timeframe} days={days}")
        filename = f"perpetual_{symbol}_mark_{timeframe}_{days}d.parquet"

        if Path(self.data_dir / filename).exists():
            df = pd.read_parquet(self.data_dir / filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.perpetual_mark_ohlcv_data[symbol] = df
            return df
        else:
            logger.info(f"Collecting perpetual mark price OHLCV for {symbol}...")

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        ccxt_symbol = self.convert_symbol_to_ccxt(symbol, "future")
        print(f"DEBUG ccxt_symbol={ccxt_symbol} filename={filename}")
        periods_per_day = self.get_periods_per_day(timeframe)
        total_periods = days * periods_per_day
        max_records_per_request = 1000
        
        all_ohlcv = []
        current_end_time = end_time
        
        num_batches = (total_periods + max_records_per_request - 1) // max_records_per_request
        
        for batch in range(num_batches):
            try:
                batch_days = min(days, max_records_per_request / periods_per_day)
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
            
            df.to_parquet(self.data_dir / filename, index=False)
            logger.info(f"  Saved perpetual mark OHLCV for {symbol}: {len(df):,} records to {filename}")
            self.perpetual_mark_ohlcv_data[symbol] = df
            return df
        else:
            logger.error(f"  No perpetual mark OHLCV data collected for {symbol}")
    
    def collect_perpetual_index_ohlcv(self, symbol, timeframe='15m', days=None ):
        """Collect perpetual index price OHLCV data"""
        filename = f"perpetual_{symbol}_index_{timeframe}_{days}d.parquet"

        if Path(self.data_dir / filename).exists():
            df = pd.read_parquet(self.data_dir / filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.perpetual_index_ohlcv_data[symbol] = df
            return df
        else:
            logger.info(f"Collecting perpetual index price OHLCV for {symbol}...")
            

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        ccxt_symbol = self.convert_symbol_to_ccxt(symbol, "future")
        periods_per_day = self.get_periods_per_day(timeframe)
        total_periods = days * periods_per_day
        max_records_per_request = 1000
        
        all_ohlcv = []
        current_end_time = end_time
        
        num_batches = (total_periods + max_records_per_request - 1) // max_records_per_request
        
        for batch in range(num_batches):
            try:
                batch_days = min(days, max_records_per_request / periods_per_day)
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
            
            df.to_parquet(self.data_dir / filename, index=False)
            logger.info(f"  Saved perpetual index OHLCV for {symbol}: {len(df):,} records to {filename}")
            self.perpetual_index_ohlcv_data[symbol] = df
            return df
        else:
            logger.error(f"  No perpetual index OHLCV data collected for {symbol}")

    
    def collect_funding_rates(self, symbol, days=None):
        """Collect funding rates data"""
        filename = f"perpetual_{symbol}_funding_rates_{days}d.parquet"

        if Path(self.data_dir / filename).exists():
            df = pd.read_parquet(self.data_dir / filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.funding_rates_data[symbol] = df
            return df
        else:
            logger.info(f"Collecting funding rates for {symbol}...")
        
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        since = int(start_time.timestamp() * 1000)
        ccxt_symbol = self.convert_symbol_to_ccxt(symbol, "future")
        try:
            funding_rates = self.futures_exchange.fetch_funding_rate_history(
                ccxt_symbol, 
                since=since,
                limit=1000,
            )
            print(funding_rates)
            
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
                
                df.to_parquet(self.data_dir / filename, index=False)
                logger.info(f"  Saved funding rates for {symbol}: {len(df):,} records to {filename}")
                self.funding_rates_data[symbol] = df
                return df
            else:
                self.funding_rates_data[symbol] = df
                logger.error(f"  No funding rate data for {symbol}")
            
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Failed to collect funding rates for {symbol}: {e}")
    
    def collect_open_interest(self, symbol, timeframe='15m', days=None):
        """Collect open interest data"""
        filename = f"perpetual_{symbol}_open_interest_{days}d_{timeframe}.parquet"

        if Path(self.data_dir / filename).exists():
            df = pd.read_parquet(self.data_dir / filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.open_interest_data[symbol] = df
            return df
        else:
            logger.info(f"Collecting open interest for {symbol}...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        ccxt_symbol = self.convert_symbol_to_ccxt(symbol, "future")
        all_open_interest = []

        periods_per_day = self.get_periods_per_day(timeframe)
        total_periods = days * periods_per_day
        max_records_per_request = 1000
        
        current_end_time = end_time
        
        num_batches = (total_periods + max_records_per_request - 1) // max_records_per_request
        
        for batch in range(num_batches):
            try:
                batch_days = min(days, max_records_per_request / periods_per_day)
                batch_start_time = current_end_time - timedelta(days=batch_days)
                since = int(batch_start_time.timestamp() * 1000)
                
                open_interest = self.futures_exchange.fetch_open_interest_history(ccxt_symbol, since=since, limit=max_records_per_request, timeframe=timeframe)
                
                if open_interest:
                    all_open_interest.extend(open_interest)
                    current_end_time = batch_start_time
                    logger.info(f"  Perpetual open interest batch {batch+1}/{num_batches}: {len(open_interest)} records")
                else:
                    break
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"  Error in perpetual open interest batch {batch+1}: {e}")
                time.sleep(1)
                continue

        
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
            
            df.to_parquet(self.data_dir / filename, index=False)
            logger.info(f"  Saved open interest for {symbol}: {len(df):,} records to {filename}")
            self.open_interest_data[symbol] = df
            return df
        else:
            logger.error(f"  No open interest data for {symbol}")
        
        time.sleep(2)
    
    def collect_perpetual_trades(self, symbol, days=None):
        """Collect perpetual trades data"""
        filename = f"perpetual_{symbol}_trades_{days}d.parquet"

        if Path(self.data_dir / filename).exists():
            df = pd.read_parquet(self.data_dir / filename)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            logger.info(f"Collecting perpetual trades for {symbol}...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        ccxt_symbol = self.convert_symbol_to_ccxt(symbol, "future")
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
            
            
            df.to_parquet(self.data_dir / filename, index=False)
            logger.info(f"  Saved perpetual trades for {symbol}: {len(df):,} records to {filename}")
            self.perpetual_trades_data[symbol] = df
            return df
        else:
            logger.error(f"  No perpetual trades data for {symbol}")
    
    def get_periods_per_day(self, timeframe):
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