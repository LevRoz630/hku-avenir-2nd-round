#!/usr/bin/env python3
f"""
Comprehensive Backtester for Trading Strategies
Compatible with existing OMS-based trading strategies with minimal code changes.

This backtester provides OMS-compatible functions that work with historical data
instead of live trading, allowing you to test strategies without modification.

Notes:
To accomodate different types of historical data, in their repsective functions of _load_()_data ensure to change the timeframes/days
how does hisotircal data work because there are different time frames, and trades at different time steps
target_value position needs to be updated because isa ctually not positions but in USDT so need to see 
margin and leverage need to be considered
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BacktesterOMS:
    """
    Fully functional OMS client for backtesting that tracks positions for perpetuals and balances for spot.
    Provides the same interface as the real OMS client but uses historical data for backtesting.
    """
    
    def __init__(self, historical_data_dir: str = "../historical_data"):
        self.historical_data_dir = Path(historical_data_dir)
        self.positions = {}  # Open Positions for Perpetuals: {symbol: {quantity, value, side, entry_price, pnl}}
        self.balance = {"USDT": 10000.0}  # Balance for Spot trading
        self.trade_history = []  # All trades executed
        self.performance_metrics = {}
        self.current_time = None  # Current backtest time
        self.data_manager = None  # Will be set by backtester
        
    def set_data_manager(self, data_manager):
        """Set the data manager for price fetching"""
        self.data_manager = data_manager
        
    def set_current_time(self, current_time: datetime):
        """Set current backtest time"""
        self.current_time = current_time
        
    def get_balance(self) -> List[Dict[str, Any]]:
        """Get account balance - compatible with OMS interface"""
        return [{"asset": asset, "balance": str(balance)} for asset, balance in self.balance.items()]
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance as dictionary - compatible with strategy interface"""
        return {asset: float(balance) for asset, balance in self.balance.items()}
    
    def get_position(self) -> List[Dict[str, Any]]:
        """Get current positions - compatible with OMS interface"""
        positions = []
        for symbol, pos in self.positions.items():
            if abs(pos['quantity']) > 0:
                # Calculate current value and PnL
                current_price = self._get_current_price(symbol)
                if current_price:
                    current_value = abs(pos['quantity']) * current_price
                    pnl = pos['quantity'] * (current_price - pos['entry_price'])
                    
                    positions.append({
                        'instrument_name': symbol,
                        'instrument_type': 'future',
                        'position_side': pos['side'],
                        'quantity': str(pos['quantity']),
                        'value': str(current_value),
                        'entry_price': str(pos['entry_price']),
                        'current_price': str(current_price),
                        'pnl': str(pnl),
                        'unrealized_pnl': str(pnl)
                    })
        return positions
    
    def set_target_position(self, instrument_name: str, instrument_type: str, 
                          target_value: float, position_side: str) -> Dict[str, Any]:
        """Set target position - compatible with OMS interface"""
        try:
            if instrument_type == "future":
                return self._set_perpetual_position(instrument_name, target_value, position_side)
            elif instrument_type == "spot":
                return self._set_spot_position(instrument_name, target_value, position_side)
            else:
                raise ValueError(f"Unsupported instrument type: {instrument_type}")
                
        except Exception as e:
            logger.error(f"Error setting target position: {e}")
            return {"id": f"error_{len(self.trade_history)}", "status": "error", "message": str(e)}
    #TODO: check which side the price is like is it btc/usdt or usdt/btc like what 
    def _set_perpetual_position(self, symbol: str, target_value: float, position_side: str) -> Dict[str, Any]:
        """Set perpetual position"""
        current_price = self._get_current_price(symbol)
        if not current_price:
            raise ValueError(f"Unable to get current price for {symbol}")
            
        # Calculate target quantity
        trade_quantity = abs(target_value / current_price)
        if position_side == "SHORT":
            trade_quantity = -trade_quantity #TODO: check pnl in posiitons what is that , yes i thnink itneeds update every iteration for position value as well as pnl (two things)
        

        # Get current position
        current_pos = self.positions.get(symbol, {'quantity': 0, 'value': 0, 'side': 'LONG', 'entry_price': 0, 'pnl': 0})
        current_quantity = current_pos['quantity']
        current_side = current_pos['side']
        current_entry_price = current_pos['entry_price']

        if position_side == current_side: # Same side perps contract e.g. buy more longs when i am already long, buy more shorts when i already short
            self.positions[symbol]['quantity'] = current_quantity + trade_quantity
            self.positions[symbol]['value'] = (current_quantity + trade_quantity) * current_price
            self.positions[symbol]['entry_price'] = ((current_entry_price * current_quantity +  current_price * trade_quantity )/ (current_quantity + trade_quantity))
        elif position_side != current_side: # Different side perps contract e.g. buy more longs when i am already short, buy more shorts when i am already long
            self.positions[symbol]['quantity'] = current_quantity + trade_quantity
            self.positions[symbol]['value'] = (current_quantity + trade_quantity) * current_price
            self.positions[symbol]['entry_price'] = ((current_entry_price * current_quantity +  current_price * trade_quantity )/ (current_quantity + trade_quantity))
            # Check if the sign of the total quantity has changed (flip long <-> short)
            if (current_quantity > 0 and current_quantity + trade_quantity < 0) or (current_quantity < 0 and current_quantity + trade_quantity > 0):
                new_side = 'SHORT' if current_quantity + trade_quantity < 0 else 'LONG'
                self.positions[symbol]['quantity'] = current_quantity + trade_quantity
                self.positions[symbol]['value'] = (current_quantity + trade_quantity) * current_price
                self.positions[symbol]['side'] = new_side
                self.positions[symbol]['entry_price'] = current_price

            if self.positions[symbol]['quantity'] == 0: #set 0
                # update balance pnl realized
                self.balance['USDT'] += self.positions[symbol]['pnl']
                self.positions[symbol] = {'quantity': 0, 'value': 0, 'side': 'LONG', 'entry_price': 0, 'pnl': 0}
       
        self.trade_history.append({
            'timestamp': self.current_time or datetime.now(),
            'symbol': symbol,
            'type': 'future',
            'side': position_side,
            'quantity': trade_quantity,
            'value': target_value,
            'price': current_price,
            'balance_after': self.balance.copy()
        })
        
        return {"id": f"backtest_{len(self.trade_history)}", "status": "success"}
    
    def _set_spot_position(self, symbol: str, target_value: float, position_side: str) -> Dict[str, Any]:
        """Set spot position"""
        # Extract base asset from symbol (e.g., BTC from BTC-USDT)
        base_asset = symbol.split('-')[0]
        
        current_price = self._get_current_price(symbol)
        if not current_price:
            raise ValueError(f"Unable to get current price for {symbol}")
        
        # Calculate target quantity
        target_quantity = target_value / current_price
        if position_side == "SHORT":
            target_quantity = -target_quantity
        
        # Get current balance
        current_balance = self.balance.get(base_asset, 0)
        
        # Calculate trade quantity
        trade_quantity = target_quantity - current_balance
        
        if abs(trade_quantity) < 1e-8:  # No significant change
            return {"id": f"no_change_{len(self.trade_history)}", "status": "success", "message": "No position change needed"}
        
        # Execute the trade
        if abs(trade_quantity) > 0:
            trade_value = abs(trade_quantity) * current_price
            
            if trade_quantity > 0:  # Buying
                if self.balance.get('USDT', 0) < trade_value:
                    raise ValueError(f"Insufficient USDT balance. Required: {trade_value}, Available: {self.balance.get('USDT', 0)}")
                self.balance['USDT'] -= trade_value
                self.balance[base_asset] = self.balance.get(base_asset, 0) + trade_quantity
            else:  # Selling
                if self.balance.get(base_asset, 0) < abs(trade_quantity):
                    raise ValueError(f"Insufficient {base_asset} balance. Required: {abs(trade_quantity)}, Available: {self.balance.get(base_asset, 0)}")
                self.balance[base_asset] -= abs(trade_quantity)
                self.balance['USDT'] += trade_value
            
            # Record trade
            self.trade_history.append({
                'timestamp': self.current_time or datetime.now(),
                'symbol': symbol,
                'type': 'spot',
                'side': position_side,
                'quantity': trade_quantity,
                'value': trade_value,
                'price': current_price,
                'balance_after': self.balance.copy()
            })
        
        return {"id": f"backtest_{len(self.trade_history)}", "status": "success"}
    
    def _get_current_price(self, symbol: str, price_type: str = 'mark') -> Optional[float]:
        """Get current price for a symbol
        
        Args:
            symbol: Trading symbol
            price_type: 'mark', 'index', or 'spot'
        """
        if not self.data_manager or not self.current_time:
            return None
            
        try:
            if price_type == 'mark':
                ohlcv_data = self.data_manager.get_data_at_time(symbol, self.current_time, 'perpetual_mark_ohlcv')
            elif price_type == 'index':
                ohlcv_data = self.data_manager.get_data_at_time(symbol, self.current_time, 'perpetual_index_ohlcv')
            elif price_type == 'spot':
                ohlcv_data = self.data_manager.get_data_at_time(symbol, self.current_time, 'spot_ohlcv')
            else:
                # Default to perpetual mark price
                ohlcv_data = self.data_manager.get_data_at_time(symbol, self.current_time, 'perpetual_mark_ohlcv')
                
            if not ohlcv_data.empty:
                return float(ohlcv_data.iloc[-1]['close'])
        except Exception as e:
            logger.error(f"Error getting current {price_type} price for {symbol}: {e}")
        
        return None
    
    def update_positions_pnl(self):
        """Update PnL for all perpetual positions"""
        for symbol, pos in self.positions.items():
            current_price = self._get_current_price(symbol)
            if current_price and pos['quantity'] != 0:
                pos['pnl'] = pos['quantity'] * (current_price - pos['entry_price'])
                pos['value'] = abs(pos['quantity']) * current_price
    
    def get_total_portfolio_value(self) -> float:
        """Get total portfolio value including positions and balances"""
        total_value = self.balance.get('USDT', 0)
        
        # Add value of perpetual positions
        for symbol, pos in self.positions.items():
            if abs(pos['quantity']) > 0:
                current_price = self._get_current_price(symbol)
                if current_price:
                    total_value += pos['quantity'] * current_price
        
        # Add value of spot balances (excluding USDT)
        for asset, balance in self.balance.items():
            if asset != 'USDT' and balance > 0:
                # Try to get price for spot asset
                spot_symbol = f"{asset}-USDT"
                current_price = self._get_current_price(spot_symbol)
                if current_price:
                    total_value += balance * current_price
        
        return total_value
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history"""
        return self.trade_history.copy()
    
    def reset_positions(self):
        """Reset all positions and balances to initial state"""
        self.positions = {}
        self.balance = {"USDT": 10000.0}
        self.trade_history = []
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get a summary of all positions and balances"""
        summary = {
            'balances': self.balance.copy(),
            'positions': {},
            'total_portfolio_value': self.get_total_portfolio_value(),
            'total_trades': len(self.trade_history)
        }
        
        for symbol, pos in self.positions.items():
            if abs(pos['quantity']) > 0:
                current_price = self._get_current_price(symbol)
                if current_price:
                    summary['positions'][symbol] = {
                        'quantity': pos['quantity'],
                        'side': pos['side'],
                        'entry_price': pos['entry_price'],
                        'current_price': current_price,
                        'pnl': pos['quantity'] * (current_price - pos['entry_price']),
                        'value': abs(pos['quantity']) * current_price
                    }
        
        return summary
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the backtest"""
        if not self.trade_history:
            return {}
        
        total_trades = len(self.trade_history)
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0
        
        for trade in self.trade_history:
            if 'pnl' in trade:
                pnl = trade['pnl'] #TODO: might not be  the best way of calcing pnl for each trade 
                total_pnl += pnl
                if pnl > 0:
                    winning_trades += 1
                elif pnl < 0:
                    losing_trades += 0
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'average_trade_pnl': total_pnl / total_trades if total_trades > 0 else 0
        }
    
    def close(self):
        """Close client - compatible with OMS interface"""
        pass

class HistoricalDataManager:
    """Manages loading and accessing historical data for backtesting"""
    
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
            
        logger.info(f"Loading comprehensive historical data for {len(symbols)} symbols...")
        
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
        for timeframe in ['1m', '5m', '15m', '1h', '1d']:
            for days in [1, 7, 30, 90]:
                filename = f"spot_{symbol}_{timeframe}_{days}d.csv"
                filepath = self.data_dir / filename
                if filepath.exists():
                    df = pd.read_csv(filepath)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df
        
        logger.warning(f"No spot OHLCV data found for {symbol}")
        return pd.DataFrame()
    
    def _load_spot_trades_data(self, symbol: str) -> pd.DataFrame:
        """Load spot trades data for a symbol"""
        for days in [1, 7, 30, 90]:
            filename = f"spot_{symbol}_trades_{days}d.csv"
            filepath = self.data_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        
        logger.warning(f"No spot trades data found for {symbol}")
        return pd.DataFrame()
    
    def _load_perpetual_mark_ohlcv_data(self, symbol: str) -> pd.DataFrame:
        """Load perpetual mark price OHLCV data for a symbol"""
        for timeframe in ['1m', '5m', '15m', '1h', '1d']:
            for days in [1, 7, 30, 90]:
                filename = f"perpetual_{symbol}_mark_{timeframe}_{days}d.csv"
                filepath = self.data_dir / filename
                if filepath.exists():
                    df = pd.read_csv(filepath)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df
        
        logger.warning(f"No perpetual mark OHLCV data found for {symbol}")
        return pd.DataFrame()
    
    def _load_perpetual_index_ohlcv_data(self, symbol: str) -> pd.DataFrame:
        """Load perpetual index price OHLCV data for a symbol"""
        for timeframe in ['1m', '5m', '15m', '1h', '1d']:
            for days in [1, 7, 30, 90]:
                filename = f"perpetual_{symbol}_index_{timeframe}_{days}d.csv"
                filepath = self.data_dir / filename
                if filepath.exists():
                    df = pd.read_csv(filepath)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df
        
        logger.warning(f"No perpetual index OHLCV data found for {symbol}")
        return pd.DataFrame()
    
    def _load_funding_data(self, symbol: str) -> pd.DataFrame:
        """Load funding rate data for a symbol"""
        for days in [1, 7, 30, 90]:
            filename = f"perpetual_{symbol}_funding_rates_{days}d.csv"
            filepath = self.data_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        
        logger.warning(f"No funding rate data found for {symbol}")
        return pd.DataFrame()
    
    def _load_open_interest_data(self, symbol: str) -> pd.DataFrame:
        """Load open interest data for a symbol"""
        for timeframe in ['5m', '15m', '1h']:
            for days in [1, 7, 30, 90]:
                filename = f"perpetual_{symbol}_open_interest_{days}d_{timeframe}.csv"
                filepath = self.data_dir / filename
                if filepath.exists():
                    df = pd.read_csv(filepath)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df
        
        logger.warning(f"No open interest data found for {symbol}")
        return pd.DataFrame()
    
    def _load_perpetual_trades_data(self, symbol: str) -> pd.DataFrame:
        """Load perpetual trades data for a symbol"""
        for days in [1, 7, 30, 90]:
            filename = f"perpetual_{symbol}_trades_{days}d.csv"
            filepath = self.data_dir / filename
            if filepath.exists():
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

class Backtester:
    """
    Main backtester class that provides OMS-compatible interface for testing strategies
    """
    
    def __init__(self, 
                 historical_data_dir: str = "historical_data",
                 initial_balance: float = 10000.0,
                 symbols: List[str] = None):
        
        self.historical_data_dir = historical_data_dir
        self.symbols = symbols 
        # or [
        #     "BTC-USDT-PERP", "ETH-USDT-PERP", "SOL-USDT-PERP", 
        #     "BNB-USDT-PERP", "XRP-USDT-PERP"
        # ]
        
        # Initialize components
        self.data_manager = HistoricalDataManager(historical_data_dir)
        self.oms_client = BacktesterOMS(historical_data_dir)
        
        # Set initial balance and connect data manager
        self.oms_client.balance = {"USDT": initial_balance}
        self.oms_client.set_data_manager(self.data_manager)
        
        # Load historical data
        self.data_manager.load_historical_data(self.symbols)
        
        # Backtesting state
        self.current_time = None
        self.start_time = None
        self.end_time = None
        self.time_step = timedelta(hours=1)  # Default 1-hour steps
        
        # Performance tracking
        self.portfolio_values = []
        self.daily_returns = []
        self.max_drawdown = 0
        self.sharpe_ratio = 0
        
        self.initial_balance = initial_balance
        
        logger.info(f"Backtester initialized with {len(self.symbols)} symbols")
    
    def get_spot_ohlcv_data(self) -> Dict[str, pd.DataFrame]:
        """Get spot OHLCV data - compatible with strategy interface"""
        data = {}
        for symbol in self.symbols:
            data[symbol] = self.data_manager.get_data_at_time(symbol, self.current_time, 'spot_ohlcv')
        return data
    
    def get_perpetual_mark_ohlcv_data(self) -> Dict[str, pd.DataFrame]:
        """Get perpetual mark price OHLCV data"""
        data = {}
        for symbol in self.symbols:
            data[symbol] = self.data_manager.get_data_at_time(symbol, self.current_time, 'perpetual_mark_ohlcv')
        return data
    
    def get_perpetual_index_ohlcv_data(self) -> Dict[str, pd.DataFrame]:
        """Get perpetual index price OHLCV data"""
        data = {}
        for symbol in self.symbols:
            data[symbol] = self.data_manager.get_data_at_time(symbol, self.current_time, 'perpetual_index_ohlcv')
        return data
    
    def get_spot_trades_data(self) -> Dict[str, pd.DataFrame]:
        """Get spot trades data"""
        data = {}
        for symbol in self.symbols:
            data[symbol] = self.data_manager.get_data_at_time(symbol, self.current_time, 'spot_trades')
        return data
    
    def get_perpetual_trades_data(self) -> Dict[str, pd.DataFrame]:
        """Get perpetual trades data"""
        data = {}
        for symbol in self.symbols:
            data[symbol] = self.data_manager.get_data_at_time(symbol, self.current_time, 'perpetual_trades')
        return data
    
    def get_funding_rate_data(self, periods: int = 10) -> Dict[str, pd.DataFrame]:
        """Get funding rate data - compatible with strategy interface"""
        data = {}
        for symbol in self.symbols:
            df = self.data_manager.get_data_at_time(symbol, self.current_time, 'funding_rates')
            if not df.empty and len(df) >= periods:
                # Get last N periods
                data[symbol] = df.tail(periods * 8).copy()  # 8 hours per period
            else:
                data[symbol] = df
        return data
    
    def get_open_interest_data(self) -> Dict[str, pd.DataFrame]:
        """Get open interest data - compatible with strategy interface"""
        data = {}
        for symbol in self.symbols:
            data[symbol] = self.data_manager.get_data_at_time(symbol, self.current_time, 'open_interest')
        return data
    
    def get_comprehensive_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Get all available data types for all symbols"""
        data = {}
        for symbol in self.symbols:
            data[symbol] = self.data_manager.get_comprehensive_data_at_time(symbol, self.current_time)
        return data
    
    # Legacy method for backward compatibility
    def get_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Get historical OHLCV data - compatible with strategy interface (legacy method)"""
        return self.get_perpetual_mark_ohlcv_data()  # Default to perpetual mark price for backward compatibility
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance - compatible with strategy interface"""
        balance_dict = {}
        for balance in self.oms_client.get_balance():
            balance_dict[balance['asset']] = float(balance['balance'])
        return balance_dict
    
    def get_current_positions(self) -> List[Dict[str, Any]]:
        """Get current positions - compatible with strategy interface"""
        return self.oms_client.get_position()
    
    def push_target_positions(self, positions: Dict[str, float], type: str = "future"):
        """Push target positions - compatible with strategy interface"""
        for symbol, target_value in positions.items():
            if abs(target_value) > 0:
                position_side = "LONG" if target_value >= 0 else "SHORT"
                self.oms_client.set_target_position(
                    instrument_name=symbol,
                    instrument_type=type,
                    target_value=abs(target_value),
                    position_side=position_side
                )
    
    def show_account_detail(self):
        """Show account details - compatible with strategy interface"""
        logger.info("==== Account Details ====")
        logger.info(f"Account Balance: {self.get_account_balance()}")
        logger.info(f"Current Positions: {self.get_current_positions()}")
    
    def run_backtest(self, 
                     strategy_class,
                     start_date: datetime = None,
                     end_date: datetime = None,
                     time_step: timedelta = None) -> Dict[str, Any]:
        """
        Run backtest with a strategy class
        
        Args:
            strategy_class: Strategy class to test
            start_date: Start date for backtest
            end_date: End date for backtest
            time_step: Time step for backtest iteration
            
        Returns:
            Dictionary with backtest results and performance metrics
        """
        
        # Set up time range
        if start_date is None:
            start_date = self._get_earliest_data_time()
        if end_date is None:
            end_date = self._get_latest_data_time()
        if time_step is None:
            time_step = timedelta(hours=1)
            
        self.start_time = start_date
        self.end_time = end_date
        self.time_step = time_step
        
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        logger.info(f"Running data from {start_date}")
        # Initialize strategy
        strategy = strategy_class()
        
        # Replace strategy's OMS functions with backtester functions
        strategy.oms_client = self.oms_client
        strategy.get_historical_data = self.get_historical_data  # Legacy method
        strategy.get_spot_ohlcv_data = self.get_spot_ohlcv_data
        strategy.get_perpetual_mark_ohlcv_data = self.get_perpetual_mark_ohlcv_data
        strategy.get_perpetual_index_ohlcv_data = self.get_perpetual_index_ohlcv_data
        strategy.get_spot_trades_data = self.get_spot_trades_data
        strategy.get_perpetual_trades_data = self.get_perpetual_trades_data
        strategy.get_funding_rate_data = self.get_funding_rate_data
        strategy.get_open_interest_data = self.get_open_interest_data
        strategy.get_comprehensive_data = self.get_comprehensive_data
        strategy.get_account_balance = self.get_account_balance
        strategy.get_current_positions = self.get_current_positions
        strategy.push_target_positions = self.push_target_positions
        strategy.show_account_detail = self.show_account_detail
        
        # Run backtest
        self.current_time = start_date
        iteration = 0
        
        while self.current_time <= end_date:
            try:
                # Update OMS client's current time
                self.oms_client.set_current_time(self.current_time)
                
                # Update strategy's current time
                strategy.current_time = self.current_time
                
                # Update positions PnL before strategy runs
                self.oms_client.update_positions_pnl()

                total_value = self.oms_client.get_total_portfolio_value()
                self.portfolio_values.append(total_value)
                summary = self.oms_client.get_position_summary()
                logger.info(f"Total Portfolio Value: {total_value}")
                logger.info(f"Position Summary: {summary}")
                # Run strategy logic
                # if hasattr(strategy, 'run_strategy'):
                strategy.run_strategy()
                # elif hasattr(strategy, 'on_1min_kline'):
                #     # For strategies that use kline callbacks
                #     for symbol in self.symbols:
                #         ohlcv_data = self.data_manager.get_data_at_time(symbol, self.current_time, 'ohlcv')
                #         if not ohlcv_data.empty:
                #             latest_candle = ohlcv_data.iloc[-1]
                #             kline = [
                #                 int(latest_candle['timestamp'].timestamp() * 1000),
                #                 latest_candle['open'],
                #                 latest_candle['high'],
                #                 latest_candle['low'],
                #                 latest_candle['close'],
                #                 latest_candle['volume']
                #             ]
                            # Note: This would need to be adapted for async strategies
                            # strategy.on_1min_kline(symbol, kline)
                
                
                # Move to next time step
                self.current_time += time_step
                iteration += 1
                
                # Log progress every 24 iterations (daily if hourly steps)
                if iteration % 24 == 0:
                    logger.info(f"Backtest progress: {self.current_time} (Iteration {iteration})")
                    
            except Exception as e:
                logger.error(f"Error in backtest iteration {iteration}: {e}")
                self.current_time += time_step
                continue
        
        # Calculate final performance metrics
        self._calculate_performance_metrics()
        
        # Return results
        return {
            'portfolio_values': self.portfolio_values,
            'daily_returns': self.daily_returns,
            'total_return': (self.portfolio_values[-1] / self.portfolio_values[0] - 1) if self.portfolio_values else 0,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'trade_history': self.oms_client.trade_history,
            'final_balance': self.get_account_balance(),
            'final_positions': self.get_current_positions()
        }
    
    def _get_earliest_data_time(self) -> datetime:
        """Get earliest timestamp from all loaded data"""
        earliest = None
        
        # Check all data types for earliest timestamp
        all_data_sources = [
            self.data_manager.spot_ohlcv_data,
            self.data_manager.spot_trades_data,
            self.data_manager.perpetual_mark_ohlcv_data,
            self.data_manager.perpetual_index_ohlcv_data,
            self.data_manager.funding_data,
            self.data_manager.open_interest_data,
            self.data_manager.perpetual_trades_data
        ]
        
        for data_source in all_data_sources:
            for symbol in self.symbols:
                data = data_source.get(symbol, pd.DataFrame())
                if not data.empty:
                    symbol_earliest = data['timestamp'].min()
                    if earliest is None or symbol_earliest < earliest:
                        earliest = symbol_earliest
        
        return earliest or datetime.now() - timedelta(days=7)
    
    def _get_latest_data_time(self) -> datetime:
        """Get latest timestamp from all loaded data"""
        latest = None
        
        # Check all data types for latest timestamp
        all_data_sources = [
            self.data_manager.spot_ohlcv_data,
            self.data_manager.spot_trades_data,
            self.data_manager.perpetual_mark_ohlcv_data,
            self.data_manager.perpetual_index_ohlcv_data,
            self.data_manager.funding_data,
            self.data_manager.open_interest_data,
            self.data_manager.perpetual_trades_data
        ]
        
        for data_source in all_data_sources:
            for symbol in self.symbols:
                data = data_source.get(symbol, pd.DataFrame())
                if not data.empty:
                    symbol_latest = data['timestamp'].max()
                    if latest is None or symbol_latest > latest:
                        latest = symbol_latest
        
        return latest or datetime.now()
    
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics"""
        if len(self.portfolio_values) < 2:
            return
            
        # Calculate daily returns
        self.daily_returns = []
        for i in range(1, len(self.portfolio_values)):
            daily_return = (self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1] # TODO: how are you sure this is daily given there might be differnt timesteps
            self.daily_returns.append(daily_return)
        
        # Calculate max drawdown
        peak = self.portfolio_values[0]
        self.max_drawdown = 0
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        
        # Calculate Sharpe ratio (simplified)
        if self.daily_returns:
            mean_return = np.mean(self.daily_returns)
            std_return = np.std(self.daily_returns)
            if std_return > 0:
                self.sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
    
    def print_results(self, results: Dict[str, Any]):
        """Print backtest results in a formatted way"""
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Final Balance: {results['final_balance']}")
        print(f"Number of Trades: {len(results['trade_history'])}")
        print("="*50)

# Example usage and strategy adapter
def create_strategy_adapter(strategy_class):
    """
    Create an adapter that makes any strategy class compatible with the backtester
    """
    class StrategyAdapter(strategy_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # The backtester will replace these methods
            pass
    
    return StrategyAdapter






# Example usage
if __name__ == "__main__":
    # Example of how to use the backtester
    from datetime import datetime, timedelta
    
    # Initialize backtester
    backtester = Backtester(
        historical_data_dir="historical_data",
        initial_balance=10000.0,
        symbols=["BTC-USDT-PERP", "ETH-USDT-PERP"]
    )
    
    # Example strategy class (replace with your actual strategy)
    class ExampleStrategy:
        def __init__(self):
            self.oms_client = None
            self.current_time = None
            self.trade_count = 0
            
        def run_strategy(self):
            # Your strategy logic here
            print(f"Running strategy at {self.current_time}")
            
            # Example: simple buy and hold with some logic
            if self.trade_count == 0:
                # Open long position in BTC perpetual
                result = self.oms_client.set_target_position(
                    instrument_name="BTC-USDT-PERP",
                    instrument_type="future",
                    target_value=1000.0,
                    position_side="LONG"
                )
                print(f"Opened BTC position: {result}")
                self.trade_count += 1
            elif self.trade_count == 1 and self.current_time.hour == 12:
                # Close position at noon
                result = self.oms_client.set_target_position(
                    instrument_name="BTC-USDT-PERP",
                    instrument_type="future",
                    target_value=0.0,
                    position_side="LONG"
                )
                print(f"Closed BTC position: {result}")
                self.trade_count += 1
    
    # Run backtest
    results = backtester.run_backtest(
        strategy_class=ExampleStrategy,
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now(),
        time_step=timedelta(hours=1)
    )
    
    # Print results
    backtester.print_results(results)
    
    # Demonstrate additional functionality
    print("\n" + "="*50)
    print("DETAILED RESULTS")
    print("="*50)
    
    # Show position summary
    position_summary = backtester.oms_client.get_position_summary()
    print(f"Final Portfolio Value: ${position_summary['total_portfolio_value']:.2f}")
    print(f"Total Trades: {position_summary['total_trades']}")
    print(f"Final Balances: {position_summary['balances']}")
    
    if position_summary['positions']:
        print("\nFinal Positions:")
        for symbol, pos in position_summary['positions'].items():
            print(f"  {symbol}: {pos['quantity']:.4f} {pos['side']} @ ${pos['entry_price']:.2f} (Current: ${pos['current_price']:.2f}, PnL: ${pos['pnl']:.2f})")
    
    # Show performance metrics
    perf_metrics = backtester.oms_client.get_performance_metrics()
    if perf_metrics:
        print(f"\nPerformance Metrics:")
        print(f"  Win Rate: {perf_metrics['win_rate']:.2%}")
        print(f"  Total PnL: ${perf_metrics['total_pnl']:.2f}")
        print(f"  Average Trade PnL: ${perf_metrics['average_trade_pnl']:.2f}")
    
    # Show recent trades
    trade_history = backtester.oms_client.get_trade_history()
    if trade_history:
        print(f"\nRecent Trades (last 5):")
        for trade in trade_history[-5:]:
            print(f"  {trade['timestamp']}: {trade['side']} {trade['quantity']:.4f} {trade['symbol']} @ ${trade['price']:.2f} (Value: ${trade['value']:.2f})")
