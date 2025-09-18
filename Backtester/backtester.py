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
        self.symbols = []  # Will be set by backtester
        self.portfolio_values = []
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

        print(f"debug positions:{self.positions}")
        # Add value of perpetual positions
        for symbol, pos in self.positions.items():
            pos_value = pos['value']
            print(f"debug pos_value:{pos_value}")
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

        
    def run_backtest(self, 
                     strategy_class: Any,
                     symbols: List[str],
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
                # Pass on all the attributes of the class to the trategy for execution
        strategy = strategy_class(symbols=symbols)

        # Pass on all the attributes of the class to the strategy for execution
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                setattr(strategy, attr_name, getattr(self, attr_name))
        print(strategy.__dict__)
        # Run backtest
        iteration = 0
        strategy.current_time = start_date
        strategy.start_time = start_date
        strategy.end_time = end_date
        strategy.time_step = time_step
        strategy.portfolio_values = []
        strategy.daily_returns = []
        strategy.max_drawdown = 0
        strategy.sharpe_ratio = 0
        strategy.trade_history = []
        strategy.final_balance = 0
        strategy.final_positions = []

        while strategy.current_time <= end_date:
            try:
                # Update OMS client's current time
                strategy.set_current_time(strategy.current_time)

                # Update strategy's current time
                strategy.current_time = strategy.current_time
                
                total_value = strategy.get_total_portfolio_value()
                print(f"debug total value:{total_value}")
                strategy.portfolio_values.append(total_value)
                summary = strategy.get_position_summary()
                logger.info(f"Total Portfolio Value: {total_value}")
                logger.info(f"Position Summary: {summary}")

                strategy.run_strategy()
               
                
                # Move to next time step
                strategy.current_time += time_step
                iteration += 1
                
                # Log progress every 24 iterations (daily if hourly steps)
                if iteration % 24 == 0:
                    logger.info(f"Backtest progress: {strategy.current_time} (Iteration {iteration})")
                    
            except Exception as e:
                logger.error(f"Error in backtest iteration {iteration}: {e}")
                strategy.current_time += time_step
                continue
        
        # Calculate final performance metrics
        strategy.calculate_performance_metrics()
        
        # Return results
        return {
            'portfolio_values': strategy.portfolio_values,
            'daily_returns': strategy.daily_returns,
            'total_return': (strategy.portfolio_values[-1] / strategy.portfolio_values[0] - 1) if strategy.portfolio_values else 0,
            'max_drawdown': strategy.max_drawdown,
            'sharpe_ratio': strategy.sharpe_ratio,
            'trade_history': strategy.trade_history,
            'final_balance': strategy.get_account_balance(),
            'final_positions': strategy.get_position()
        }

    def calculate_performance_metrics(self):
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

