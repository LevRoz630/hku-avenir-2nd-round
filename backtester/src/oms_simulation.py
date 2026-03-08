#!/usr/bin/env python3
"""OMS simulation layer for backtesting. Tracks positions, balances, and trade execution."""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import json
from hist_data import HistoricalDataCollector


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OMSClient:
    
    def __init__(self, historical_data_dir: str = "../hku-data/test_data"):
        self.historical_data_dir = Path(historical_data_dir)
        self.positions = {}  # Open Positions for Perpetuals: {symbol: {quantity, value, side, entry_price, pnl}}
        self.balance = {"USDT": 10000.0}  # Balance for trading (always dict format)
        self.trade_history = []  # All trades executed
        self.current_time = None  # Current backtest time
        self.data_manager = None  # Will be set by backtester
        
    def set_current_time(self, current_time: datetime):
        self.current_time = current_time

    def set_data_manager(self, data_manager: HistoricalDataCollector):
        self.data_manager = data_manager

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.replace('-PERP', '')
    
    def get_position(self) -> List[Dict[str, Any]]:
        positions = []
        for symbol, pos in self.positions.items():

            if abs(pos['quantity']) > 0:
                # Calculate current value and PnL
                current_price = self.get_current_price(symbol)
                if current_price:
                    current_value = abs(pos['quantity']) * current_price
                    # Side-aware PnL for display
                    pnl = (
                        pos['quantity'] * (current_price - pos['entry_price'])
                        if pos.get('side') == 'LONG' else
                        pos['quantity'] * (pos['entry_price'] - current_price)
                    )
                    
                    positions.append({
                        'symbol': symbol,
                        'instrument_type': 'future',
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
        try:
            if instrument_type and instrument_type != "future":
                raise ValueError(f"Unsupported instrument type: {instrument_type}")

            return self._set_position(symbol, target_value, position_side)
                
        except Exception as e:
            logger.error(f"Error setting target position: {e}")
 
    def _set_position(self, symbol: str, trade_amount_usdt: float, position_side: str) -> Dict[str, Any]:
        current_price = self.get_current_price(symbol)
        if not current_price:
            raise ValueError(f"Unable to get current price for {symbol}")

        trade_value = float(trade_amount_usdt)
        if trade_value > self.balance['USDT']:
            raise ValueError(f"Insufficient USDT balance. Required: {trade_value}, Available: {self.balance['USDT']}")

        trade_qty = trade_value / current_price
        pos = self.positions.setdefault(symbol, {
            'quantity': 0.0,
            'value': 0.0,
            'side': 'LONG',
            'entry_price': 0.0,
            'pnl': 0.0,
            'instrument_type': 'future',
        })

        current_quantity = float(pos.get('quantity', 0.0))
        current_side = pos.get('side', 'LONG')
        current_entry_price = float(pos.get('entry_price', 0.0))

        # Handle close
        if position_side == 'CLOSE':
            # Realize PnL and zero out
            pnl, principal = self.close_position(symbol)
            self.balance['USDT'] += (pnl or 0.0) + (principal or 0.0)
            pos['pnl'] = pos.get('pnl', 0.0) + (pnl or 0.0)
            self.positions[symbol] = {
                'quantity': 0.0,
                'value': 0.0,
                'side': 'CLOSE',
                'entry_price': 0.0,
                'pnl': pos.get('pnl', 0.0),
                'instrument_type': 'future',
            }

        elif position_side == current_side:
            self.balance['USDT'] -= trade_value
            new_qty = current_quantity + trade_qty
            self.positions[symbol] = {
                'quantity': new_qty,
                'value': abs(new_qty) * current_price,
                'side': position_side,
                'entry_price': (
                    (current_entry_price * current_quantity + current_price * trade_qty) / new_qty
                ) if new_qty else current_price,
                'pnl': pos.get('pnl', 0.0),
                'instrument_type': 'future',
            }

        elif position_side != current_side:
            pnl, principal = self.close_position(symbol)
            self.balance['USDT'] += (pnl or 0.0) + (principal or 0.0)
            self.balance['USDT'] -= trade_value
            self.positions[symbol] = {
                'quantity': trade_qty,
                'value': abs(trade_qty) * current_price,
                'side': position_side,
                'entry_price': current_price,
                'pnl': pos.get('pnl', 0.0) + (pnl or 0.0),
                'instrument_type': 'future',
            }
        else:
            raise ValueError(f"Invalid position side: {position_side}")
       
        # Record trade
        self.trade_history.append({
            'timestamp': self.current_time,
            'symbol': symbol,
            'type': 'future',
            'side': self.positions[symbol]['side'],
            'quantity': self.positions[symbol]['quantity'] ,
            'value': self.positions[symbol]['value'] ,
            'price': self.positions[symbol]['entry_price'],
            'balance_after': self.balance
        })
        
        return {"id": f"backtest_{len(self.trade_history)}", "status": "success"}
    
    def get_current_price(self, symbol: str, instrument_type: str = None) -> Optional[float]:
        if not self.data_manager or not self.current_time:
            return None

        try:
            base_symbol = self._normalize_symbol(symbol)
            data_type = "mark_ohlcv_futures"

            df = self._get_preloaded_price_data(base_symbol, data_type)
            if df is not None and not df.empty:
                # Ensure timestamps are properly normalized for comparison
                if df['timestamp'].dt.tz is None:
                    # Assume UTC if tz-naive
                    df = df.copy()
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')

                # Price at or before current_time
                mask = df['timestamp'] <= self.current_time
                if mask.any():
                    price = float(df.loc[mask, 'close'].iloc[-1])
                    return price
                else:
                    logger.debug(f"No data found for {base_symbol} at or before {self.current_time}. Data time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            # Fallback: load from disk
            logger.debug(f"Loading price data for {symbol} ({base_symbol}) from disk")
            df = self.data_manager.load_data_period(base_symbol, "15m", data_type, self.current_time, self.current_time + timedelta(minutes=15), load_from_class=True)
            if df is None or df.empty:
                return None
            price = float(df['close'].iloc[-1])
            return price
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def _get_preloaded_price_data(self, symbol: str, data_type: str) -> Optional[pd.DataFrame]:
        """Get preloaded price data from memory cache."""
        try:
            # Use the same data store mapping as load_from_class
            if data_type == "mark_ohlcv_futures":
                df = self.data_manager.perpetual_mark_ohlcv_data.get(symbol)
            elif data_type == "index_ohlcv_futures":
                df = self.data_manager.perpetual_index_ohlcv_data.get(symbol)
            else:
                return None

            return df if df is not None and not df.empty else None
        except Exception:
            return None
    
    def close_position(self, symbol: str):
        """Update PnL for all perpetual positions"""
        current_price = self.get_current_price(symbol)
        pos = self.positions[symbol]
        pnl = 0.0
        principal = 0.0
        if current_price and pos['quantity'] != 0:
            if pos['side'] == 'LONG':
                pnl = pos['quantity'] * (current_price - pos['entry_price']) - 0.00023* pos['quantity'] * pos['entry_price']
            elif pos['side'] == 'SHORT':
                pnl = pos['quantity'] * (pos['entry_price'] - current_price) - 0.00023* pos['quantity'] * pos['entry_price']
            pos['value'] = abs(pos['quantity']) * current_price
            principal = pos['quantity'] * pos['entry_price']
        return pnl, principal


    def update_portfolio_value(self) -> float:
        """
        Update and return total portfolio value including balances and appropriate position valuation.
        
        """
        total_value = self.balance['USDT']

        for symbol, pos in self.positions.items():
            current_price = self.get_current_price(symbol)
            if not current_price:
                continue
            quantity = float(pos.get('quantity', 0.0))
            side = pos.get('side', 'LONG')
            entry = float(pos.get('entry_price', 0.0))

            if abs(quantity) > 0:
                if side == 'LONG':
                    unrealized = quantity * (current_price - entry)
                elif side == 'SHORT':
                    unrealized = quantity * (entry - current_price)
                else:
                    unrealized = 0.0
                total_value += unrealized

                        # Maintain notional value field for reporting only
            
            position_value = abs(quantity) * entry
            total_value += position_value
            
            pos['value'] = abs(quantity) * current_price
            
        return total_value

    def get_position_summary(self) -> Dict[str, Any]:
        """Get a summary of all positions and balances"""
        summary = {
            'balances': self.balance,
            'positions': {},
            'total_portfolio_value': self.update_portfolio_value(),
            'total_trades': len(self.trade_history)
        }
        
        for symbol, pos in self.positions.items():
            if abs(pos['quantity']) > 0:
                current_price = self.get_current_price(symbol)
                if current_price:
                    qty = float(pos['quantity'])
                    entry = float(pos['entry_price'])
                    unrealized = qty * ((current_price - entry) if pos['side'] == 'LONG' else (entry - current_price))
                    summary['positions'][symbol] = {
                            'quantity': qty,
                            'side': pos['side'],
                            'entry_price': entry,
                            'current_price': current_price,
                            'unrealized_pnl': unrealized,
                            'value': abs(qty) * current_price
                        }
        
        return summary

