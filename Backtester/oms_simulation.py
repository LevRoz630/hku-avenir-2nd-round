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
                print(f"debug setting target position for future:{symbol}")
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
        print(f"debug current price:{current_price}")
        if not current_price:
            raise ValueError(f"Unable to get current price for {symbol}")
            
        # Calculate target quantity
        trade_value = trade_quantity * current_price
        print(f"debug trade value:{trade_value}")
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
        print(f"debug getting current price for {symbol} with instrument type:{instrument_type}")
        base_symbol = symbol.replace('-PERP', '') if instrument_type in ('future', 'mark') else symbol

        try:
            if instrument_type == 'mark':
                ohlcv_data = self.data_manager.get_data_at_time(base_symbol, self.current_time, 'perpetual_mark_ohlcv')
            elif instrument_type == 'index':
                ohlcv_data = self.data_manager.get_data_at_time(base_symbol, self.current_time, 'perpetual_index_ohlcv')
            elif instrument_type == 'spot':
                ohlcv_data = self.data_manager.get_data_at_time(base_symbol, self.current_time, 'spot_ohlcv')
            elif instrument_type == 'future':
                ohlcv_data = self.data_manager.get_data_at_time(base_symbol, self.current_time, 'perpetual_mark_ohlcv')
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

