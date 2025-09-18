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
from hist_data import HistoricalDataManager


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
            instrument_type = 'future' if symbol.endswith('-PERP') else 'spot'

            if abs(pos['quantity']) > 0:
                # Calculate current value and PnL
                current_price = self._get_current_price(symbol, instrument_type)
                if current_price:
                    current_value = abs(pos['quantity']) * current_price
                    pnl = pos['quantity'] * (current_price - pos['entry_price'])
                    
                    positions.append({
                        'symbol': symbol,
                        'instrument_type': instrument_type,
                        'position_side': pos['side'],
                        'quantity': str(pos['quantity']),
                        'value': str(current_value),
                        'entry_price': str(pos['entry_price']),
                        'current_price': str(current_price),
                        'pnl': str(pnl),
                    })
        return positions
    
    def set_target_position(self, symbol: str, instrument_type: str, 
                          target_value: float, position_side: str) -> Dict[str, Any]:
        """Set target position - compatible with OMS interface"""
        try:
            if symbol.endswith("-PERP"):
                instrument_type = "future"
            elif symbol.endswith("-USDT"):
                instrument_type = "spot"
            else:
                raise ValueError(f"Invalid symbol: {symbol}")

            if instrument_type == "future":
                return self._set_position(symbol, target_value, position_side, instrument_type)
            elif instrument_type == "spot":
                return self._set_position(symbol, target_value, position_side, instrument_type)
            else:
                raise ValueError(f"Unsupported instrument type: {instrument_type}")
                
        except Exception as e:
            logger.error(f"Error setting target position: {e}")
            return {"id": f"error_{len(self.trade_history)}", "status": "error", "message": str(e)}
 
    def _set_position(self, symbol: str, trade_quantity: float, position_side: str, instrument_type: str) -> Dict[str, Any]:
        """Set perpetual position"""
        current_price = self._get_current_price(symbol, instrument_type)
        if not current_price:
            raise ValueError(f"Unable to get current price for {symbol}")
            
        # Calculate target quantity
        trade_value = trade_quantity * current_price
        if trade_value > self.balance['USDT']:
            raise ValueError(f"Insufficient USDT balance. Required: {trade_value}, Available: {self.balance['USDT']}")

        # Get current position
        current_pos = self.positions.get(symbol, {'quantity': 0, 'value': 0, 'side': 'LONG', 'entry_price': 0, 'pnl': 0})
        current_quantity = current_pos['quantity']
        current_side = current_pos['side']
        current_entry_price = current_pos['entry_price']

        if position_side == current_side:
            self.balance['USDT'] -= trade_value
            

            self.positions[symbol]['quantity'] = current_quantity + trade_quantity
            self.positions[symbol]['value'] = (current_quantity + trade_quantity) * current_price
            self.positions[symbol]['entry_price'] = ((current_entry_price * current_quantity +  current_price * trade_quantity )/ (current_quantity + trade_quantity))
            

        elif position_side != current_side:
            # calculates the profit from closing the position and reduce balance by the new position
            self.positions[symbol]['pnl'] += self.calculate_positions_pnl(symbol)
            self.balance['USDT'] += self.calculate_positions_pnl(symbol)
            self.balance['USDT'] -= trade_value
            # update the position itself
            self.positions[symbol]['quantity'] =  trade_quantity
            self.positions[symbol]['value'] = self.positions[symbol]['quantity'] * current_price
            self.positions[symbol]['side'] = position_side
            self.positions[symbol]['entry_price'] = current_price

        elif position_side == 'CLOSE':
                # returns the cash to our balance as we are closing the position
                self.balance['USDT'] += self.calculate_positions_pnl(symbol)
                self.positions[symbol]['pnl'] += self.calculate_positions_pnl(symbol)

                self.positions[symbol] = {'quantity': 0, 'value': 0, 'side': 'CLOSE', 'entry_price': 0}
        else:
            raise ValueError(f"Invalid position side: {position_side}")
       
        self.trade_history.append({
            'timestamp': self.current_time or datetime.now(),
            'symbol': symbol,
            'type': 'future',
            'side': self.positions[symbol]['side'],
            'quantity': self.positions[symbol]['quantity'] ,
            'value': self.positions[symbol]['value'] ,
            'price': self.positions[symbol]['entry_price'],
            'balance_after': self.balance.copy()
        })
        
        return {"id": f"backtest_{len(self.trade_history)}", "status": "success"}
    
    def _get_current_price(self, symbol: str, instrument_type: str = 'mark') -> Optional[float]:
        """Get current price for a symbol
        
        Args:
            symbol: Trading symbol
            instrument_type: 'mark', 'index', or 'spot', 'future'
        """
        if not self.data_manager or not self.current_time:
            return None
            
        try:
            if instrument_type == 'mark':
                ohlcv_data = self.data_manager.get_data_at_time(symbol, self.current_time, 'perpetual_mark_ohlcv')
            elif instrument_type == 'index':
                ohlcv_data = self.data_manager.get_data_at_time(symbol, self.current_time, 'perpetual_index_ohlcv')
            elif instrument_type == 'spot':
                ohlcv_data = self.data_manager.get_data_at_time(symbol, self.current_time, 'spot_ohlcv')
            elif instrument_type == 'future':
                ohlcv_data = self.data_manager.get_data_at_time(symbol, self.current_time, 'perpetual_mark_ohlcv')
            else:
                raise ValueError(f"Invalid price type: {instrument_type}")
                
            if not ohlcv_data.empty:
                return float(ohlcv_data.iloc[-1]['close'])
        except Exception as e:
            logger.error(f"Error getting current {instrument_type} price for {symbol}: {e}")
        
        return None
    
    def calculate_positions_pnl(self, symbol: str, instrument_type: str):
        """Update PnL for all perpetual positions"""
        current_price = self._get_current_price(symbol, instrument_type)
        pos = self.positions[symbol]
        if current_price and pos['quantity'] != 0:
            if pos['side'] == 'LONG':
                pnl = pos['quantity'] * (current_price - pos['entry_price'])
            elif pos['side'] == 'SHORT':
                pnl = pos['quantity'] * (pos['entry_price'] - current_price)
            pos['value'] = pos['quantity'] * current_price
        return pnl

    def get_total_portfolio_value(self) -> float:
        """Get total portfolio value including positions and balances"""
        total_value = self.balance.get('USDT', 0)
        print(f"debug total value get_total_portfolio_value:{total_value}")

        # Add value of perpetual positions
        for symbol, pos in self.positions.items():
            pos_value = pos['value']
            total_value += pos_value
        return total_value  

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
                current_price = self._get_current_price(symbol, pos['instrument_type'])
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
    
    def push_target_positions(self, positions: Dict[str, float], instrument_type: str):
        """Push target positions - compatible with strategy interface
        
        Args:
            positions: Target position dictionary {symbol: target_value(USDT)}
            instrument_type: Instrument type (future, spot)
        """
        if instrument_type != 'future' and instrument_type != 'spot':
            raise ValueError(f"Invalid instrument type: {instrument_type}")
        
        for symbol, target_value in positions.items():
            if abs(target_value) > 0:
                position_side = "LONG" if target_value >= 0 else "SHORT"
                self.oms_client.set_target_position(
                    symbol=symbol,
                    instrument_type=instrument_type,
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
        strategy = strategy_class(self)
        
        # Run backtest
        self.current_time = start_date
        iteration = 0
        
        while self.current_time <= end_date:
            try:
                # Update OMS client's current time
                self.oms_client.set_current_time(self.current_time)
                
                # Update strategy's current time
                strategy.current_time = self.current_time
                
                total_value = self.oms_client.get_total_portfolio_value()
                print(f"debug total value:{total_value}")
                self.portfolio_values.append(total_value)
                summary = self.oms_client.get_position_summary()
                logger.info(f"Total Portfolio Value: {total_value}")
                logger.info(f"Position Summary: {summary}")

                strategy.run_strategy()
               
                
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
        self.period_returns = []
        for i in range(1, len(self.portfolio_values)):
            period_return = (self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1] #
            self.period_returns.append(period_return)
        
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
