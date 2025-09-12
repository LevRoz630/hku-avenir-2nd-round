"""
Data Collection and Storage Module for Testing
Collects and saves various data types for backtesting and analysis
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import logging
from ccxt import binance as binance_sync
from sdk.oms_client import OmsClient
from config import setup_environment

logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, data_dir="data", mode="test"):
        """
        Initialize data collector
        
        Args:
            data_dir: Directory to save collected data
            mode: 'test' or 'prod' - determines OMS credentials to use
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.mode = mode
        
        # Setup environment for OMS
        try:
            setup_environment(mode)
            logger.info(f"Environment configured for {mode} mode")
        except Exception as e:
            logger.warning(f"Failed to setup environment: {e}")
        
        # Initialize exchange
        self.exchange = binance_sync({
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # Initialize OMS client
        try:
            self.oms_client = OmsClient()
            logger.info("OMS client initialized successfully")
        except Exception as e:
            self.oms_client = None
            logger.warning(f"OMS client not available: {e}")
    
    def collect_historical_data(self, symbols=None, days=7, timeframe='15m'):
        """Collect historical OHLCV data with configurable granularity and length"""
        if symbols is None:
            symbols = ["BTC-USDT-PERP", "ETH-USDT-PERP", "SOL-USDT-PERP", 
                      "BNB-USDT-PERP", "XRP-USDT-PERP"]
        
        data = {}
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        since = int(start_time.timestamp() * 1000)
        
        # Calculate appropriate limit based on timeframe
        if timeframe == '1h':
            limit = days * 24  # 24 hours per day
        elif timeframe == '15m':
            limit = days * 96  # 96 periods per day (24*4)
        elif timeframe == '1m':
            limit = days * 1440  # 1440 periods per day (24*60)
        else:
            limit = days * 96  # default to 15-minute
        
        for symbol in symbols:
            try:
                base = symbol.split('-')[0]
                ccxt_symbol = f"{base}/USDT"
                
                ohlcv = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=since, limit=limit)
                
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['symbol'] = symbol
                    data[symbol] = df
                    
                    # Save individual symbol data
                    df.to_csv(self.data_dir / f"{symbol}_{timeframe}_{days}d.csv", index=False)
                    logger.info(f"Saved {symbol}: {len(df)} records ({timeframe} for {days} days)")
                
            except Exception as e:
                logger.error(f"Failed to collect {symbol}: {e}")
        
        # Save combined data
        if data:
            combined_df = pd.concat(data.values(), ignore_index=True)
            combined_df.to_csv(self.data_dir / f"combined_historical_{timeframe}_{days}d.csv", index=False)
            
            # Save as pickle for faster loading
            with open(self.data_dir / f"historical_data_{timeframe}_{days}d.pkl", 'wb') as f:
                pickle.dump(data, f)
        
        return data
    
    def collect_account_data(self):
        """Collect account balance and position data"""
        account_data = {
            'timestamp': datetime.now().isoformat(),
            'balances': [],
            'positions': [],
            'asset_changes': []
        }
        
        if self.oms_client:
            try:
                # Get balances
                balances = self.oms_client.get_balance()
                account_data['balances'] = balances
                
                # Get positions
                positions = self.oms_client.get_position()
                account_data['positions'] = positions
                
                # Get asset changes
                changes = self.oms_client.get_asset_changes()
                account_data['asset_changes'] = changes
                
                logger.info("Account data collected successfully")
                
            except Exception as e:
                logger.error(f"Failed to collect account data: {e}")
        
        # Save account data
        with open(self.data_dir / f"account_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(account_data, f, indent=2, default=str)
        
        return account_data
    
    def calculate_returns(self, data, days=7):
        """Calculate returns for backtesting"""
        returns = {}
        
        for symbol, df in data.items():
            if len(df) < 2:
                returns[symbol] = 0
                continue
            
            start_price = df.iloc[0]['close']
            end_price = df.iloc[-1]['close']
            returns[symbol] = (end_price - start_price) / start_price
        
        # Save returns data
        returns_df = pd.DataFrame(list(returns.items()), columns=['symbol', 'return'])
        returns_df['timestamp'] = datetime.now()
        returns_df.to_csv(self.data_dir / f"returns_{days}d.csv", index=False)
        
        return returns

    def collect_all_data(self):
        """Collect all available data types"""
        logger.info("Starting comprehensive data collection...")
        
        # Historical data
        historical = self.collect_historical_data()
        
        # Account data
        account = self.collect_account_data()
        
        # Calculate returns
        returns = self.calculate_returns(historical)
        
        # Summary
        summary = {
            'collection_time': datetime.now().isoformat(),
            'symbols_collected': list(historical.keys()),
            'total_records': sum(len(df) for df in historical.values()),
            'account_available': self.oms_client is not None,
            'data_directory': str(self.data_dir)
        }
        
        with open(self.data_dir / "collection_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Data collection complete. Files saved to: {self.data_dir}")
        return summary

if __name__ == "__main__":
    collector = DataCollector()
    summary = collector.collect_all_data()
    print("Data collection summary:", summary)
