#!/usr/bin/env python3
"""
Historical Data Collector - Fetches comprehensive historical data for multiple days
Collects: OHLCV, funding rates, open interest, trades, and more for extended periods
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HistoricalDataCollector:
    def __init__(self, data_dir="historical_data", days=None):
        """Initialize historical data collector"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.days = days
        
        # Initialize Binance exchange
        self.exchange = binance_sync({
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        logger.info(f"Historical Data Collector initialized for {days} days")
    
    def collect_historical_ohlcv(self, symbols, timeframe='15m'):
        """Collect historical OHLCV data for multiple days"""
        logger.info(f"Collecting historical OHLCV data ({timeframe}) for {len(symbols)} symbols...")
        
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
        
        for symbol in symbols:
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
                        
                        ohlcv = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=since, limit=max_records_per_request)
                        
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
                    filename = f"{symbol}_{timeframe}_{self.days}d.csv"
                    df.to_csv(self.data_dir / filename, index=False)
                    logger.info(f"  Saved {symbol}: {len(df):,} records to {filename}")
                else:
                    logger.warning(f"  No data collected for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to collect {symbol}: {e}")
    
    def collect_historical_funding_rates(self, symbols):
        """Collect historical funding rates for multiple days"""
        logger.info(f"Collecting historical funding rates for {len(symbols)} symbols...")
        
        for symbol in symbols:
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
                    filename = f"{symbol}_funding_rates_{self.days}d.csv"
                    df.to_csv(self.data_dir / filename, index=False)
                    logger.info(f"  Saved {symbol}: {len(df):,} funding rate records to {filename}")
                else:
                    logger.warning(f"  No funding rate data for {symbol}")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Failed to collect funding rates for {symbol}: {e}")

    def collect_historical_open_interest(self, symbols, timeframe='5m'):
        """Collect historical open interest data with proper API rate limiting"""
        logger.info(f"Collecting historical open interest for {len(symbols)} symbols...")
        
        for symbol in symbols:
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

                    print(start_time)
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
                    if timeframe == '5m':
                        start_time = start_time + timedelta(minutes=5 * batch_size)
                    elif timeframe == '15m':
                        start_time = start_time + timedelta(minutes=15 * batch_size)
                    else:
                        print("timeframe not alligned with time shift, go fix it")
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
                    filename = f"{symbol}_open_interest_{self.days}d_{timeframe}.csv"
                    df.to_csv(self.data_dir / filename, index=False)
                    logger.info(f"  Saved {symbol}: {len(df):,} open interest records to {filename}")
                else:
                    logger.warning(f"  No open interest data for {symbol}")
                
                # Longer delay between symbols
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to collect open interest for {symbol}: {e}")

    def collect_historical_trades(self, symbols):
        """Collect historical trades data (limited to recent period due to API limits)"""
        logger.info(f"Collecting historical trades for {len(symbols)} symbols (last {self.days} days)...")
        
        for symbol in symbols:
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
                    filename = f"{symbol}_trades_{self.days}d.csv"
                    df.to_csv(self.data_dir / filename, index=False)
                    logger.info(f"  Saved {symbol}: {len(df):,} trade records to {filename}")
                else:
                    logger.warning(f"  No trades data for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to collect trades for {symbol}: {e}")

    def collect_all_historical_data(self, symbols, timeframes=['15m'], include_trades=True):
        """Collect all types of historical data"""
        logger.info("Starting comprehensive historical data collection...")
        
        start_time = datetime.now()
        
        # Collect OHLCV data for different timeframes
        for timeframe in timeframes:
            logger.info(f"\n=== Collecting OHLCV data ({timeframe}) ===")
            self.collect_historical_ohlcv(symbols, timeframe)
        
        # Collect funding rates
        logger.info(f"\n=== Collecting Funding Rates ===")
        self.collect_historical_funding_rates(symbols)
        
        # Collect open interest
        logger.info(f"\n=== Collecting Open Interest ===")
        self.collect_historical_open_interest(symbols, timeframe)
        
        # Collect trades (limited period)
        if include_trades:
            logger.info(f"\n=== Collecting Trades Data ===")
            self.collect_historical_trades(symbols)
        
        # Create summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        summary = {
            'collection_start': start_time.isoformat(),
            'collection_end': end_time.isoformat(),
            'duration_minutes': duration.total_seconds() / 60,
            'symbols_processed': len(symbols),
            'timeframes_collected': timeframes,
            'days_covered': self.days,
            'files_created': len(list(self.data_dir.glob('*.csv'))),
            'data_directory': str(self.data_dir)
        }
        
        # Save summary
        with open(self.data_dir / f"historical_collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"\n=== Collection Complete ===")
        logger.info(f"Duration: {duration}")
        logger.info(f"Files created: {summary['files_created']}")
        logger.info(f"Data directory: {self.data_dir}")
        
        return summary
