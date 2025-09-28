#!/usr/bin/env python3
"""
OMS-compatible simulation layer used by the backtester.
Provides order/position operations (spot and perpetual), portfolio valuation,
and a consistent interface for strategies to trade against historical data.
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
from hist_data import HistoricalDataCollector


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BacktesterOMS:
    """Order Management System abstraction for backtests.

    - Tracks balances and symbol positions
    - Converts target USDT to quantity at current price
    - Records trade history and realized PnL on close/flip
    - Serves prices via HistoricalDataCollector with per-timestep caching
    """
    
    def __init__(self, historical_data_dir: str = "../hku-data/test_data"):
        self.historical_data_dir = Path(historical_data_dir)
        self.positions = {}  # Open Positions for Perpetuals: {symbol: {quantity, value, side, entry_price, pnl}}
        self.balance = {"USDT": 10000.0}  # Balance for Spot trading
        self.trade_history = []  # All trades executed
        self.performance_metrics = {}
        self.current_time = None  # Current backtest time
        self.data_manager = HistoricalDataCollector(historical_data_dir)  # Will be set by backtester
        # Simple per-timestamp price cache: { (symbol, instrument_type): price }
        self._price_cache_time: Optional[datetime] = None
        self._price_cache: Dict[Tuple[str, str], float] = {}
        
    def set_current_time(self, current_time: datetime):
        """Set current backtest time"""
        self.current_time = current_time
        # Invalidate per-timestep price cache whenever the clock advances
        self._price_cache_time = current_time
        self._price_cache.clear()

    def _normalize_symbol(self, symbol: str, instrument_type: str) -> str:
        """Map trading symbol to data key used by HistoricalDataManager."""
        # Futures files are stored under base symbols; strip the -PERP suffix for lookups
        if instrument_type == "future":
            return symbol.replace('-PERP', '')
        return symbol
        
    def get_balance(self) -> List[Dict[str, Any]]:
        """Get account balance - compatible with OMS interface"""
        return [{"asset": asset, "balance": str(balance)} for asset, balance in self.balance.items()]
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance as dictionary - compatible with strategy interface"""
        return {asset: float(balance) for asset, balance in self.balance.items()}
    
    async def get_position(self) -> List[Dict[str, Any]]:
        """Return current non-zero positions with live valuation and PnL."""
        positions = []
        for symbol, pos in self.positions.items():
            instrument_type = 'future' if symbol.endswith('-PERP') else 'spot'

            if abs(pos['quantity']) > 0:
                # Calculate current value and PnL
                current_price = await self.get_current_price(symbol, instrument_type)
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
    
    async def set_target_position(self, symbol: str, instrument_type: str, 
                          target_value: float, position_side: str) -> Dict[str, Any]:
        """Place/adjust a position using target USDT value; supports CLOSE to exit."""
        try:

            if instrument_type == "future":
                return await self._set_position(symbol, target_value, position_side, instrument_type)
            elif instrument_type == "spot":
                return await self._set_position(symbol, target_value, position_side, instrument_type)
            else:
                raise ValueError(f"Unsupported instrument type: {instrument_type}")
                
        except Exception as e:
            logger.error(f"Error setting target position: {e}")
            return {"id": f"error_{len(self.trade_history)}", "status": "error", "message": str(e)}
 
    async def _set_position(self, symbol: str, trade_amount_usdt: float, position_side: str, instrument_type: str) -> Dict[str, Any]:
        """Set position using USDT amount; converts to quantity at current price"""
        current_price = await self.get_current_price(symbol, instrument_type)
        if not current_price:
            raise ValueError(f"Unable to get current price for {symbol}")

        # Interpret incoming target_value as an amount of USDT to deploy
        trade_value = float(trade_amount_usdt)
        is_futures = instrument_type == 'future'
        if not is_futures and trade_value > self.balance['USDT']:
            raise ValueError(f"Insufficient USDT balance. Required: {trade_value}, Available: {self.balance['USDT']}")

        # Convert USDT → base-asset quantity at current mark/index price
        trade_qty = trade_value / current_price

        # Ensure a position object exists for this symbol
        pos = self.positions.setdefault(symbol, {
            'quantity': 0.0,
            'value': 0.0,
            'side': 'LONG',
            'entry_price': 0.0,
            'pnl': 0.0,
            'instrument_type': instrument_type,
        })

        current_quantity = float(pos.get('quantity', 0.0))
        current_side = pos.get('side', 'LONG')
        current_entry_price = float(pos.get('entry_price', 0.0))

        # Handle explicit close before side comparisons to avoid misrouting CLOSE
        if position_side == 'CLOSE':
            # Explicit close: realize PnL and zero out the position
            pnl = await self.pnl_close_position(symbol, instrument_type)
            self.balance['USDT'] += (pnl or 0.0)
            pos['pnl'] = pos.get('pnl', 0.0) + (pnl or 0.0)
            self.positions[symbol] = {
                'quantity': 0.0,
                'value': 0.0,
                'side': 'CLOSE',
                'entry_price': 0.0,
                'pnl': pos.get('pnl', 0.0),
                'instrument_type': instrument_type,
            }

        elif position_side == current_side:
            # Add to an existing position on the same side, update VWAP entry_price
            if not is_futures:
                self.balance['USDT'] -= trade_value
            new_qty = current_quantity + trade_qty
            pos['quantity'] = new_qty
            pos['value'] = abs(new_qty) * current_price
            pos['entry_price'] = (
                (current_entry_price * current_quantity + current_price * trade_qty) / new_qty
            ) if new_qty else current_price
            pos['side'] = position_side

        elif position_side != current_side:
            # Flip side: realize PnL on the old side, then open a fresh position
            pnl = await self.pnl_close_position(symbol, instrument_type)
            pos['pnl'] = pos.get('pnl', 0.0) + (pnl or 0.0)
            self.balance['USDT'] += (pnl or 0.0)
            if not is_futures:
                self.balance['USDT'] -= trade_value
            pos['quantity'] = trade_qty
            pos['value'] = abs(trade_qty) * current_price
            pos['side'] = position_side
            pos['entry_price'] = current_price
            pos['instrument_type'] = instrument_type
        else:
            raise ValueError(f"Invalid position side: {position_side}")
       
        # Persist an immutable record of the action with post-trade state
        self.trade_history.append({
            'timestamp': self.current_time or datetime.now(),
            'symbol': symbol,
            'type': instrument_type,
            'side': self.positions[symbol]['side'],
            'quantity': self.positions[symbol]['quantity'] ,
            'value': self.positions[symbol]['value'] ,
            'price': self.positions[symbol]['entry_price'],
            'balance_after': self.balance.copy()
        })
        
        return {"id": f"backtest_{len(self.trade_history)}", "status": "success"}
    
    async def get_current_price(self, symbol: str, instrument_type: str = None) -> Optional[float]:
        """Get current price for a symbol
        
        Args:
            symbol: Trading symbol
            instrument_type: 'spot', 'future'
        """
        if not self.data_manager or not self.current_time:
            return None

        # Serve from cache if available for this backtest timestamp
        cache_key = (symbol, instrument_type)
        if self._price_cache_time == self.current_time and cache_key in self._price_cache:
            return self._price_cache[cache_key]

        try:
            # Normalize to data keys (files are stored under base symbols like BTC-USDT)
            base_symbol = self._normalize_symbol(symbol, instrument_type)
            if instrument_type == 'spot':
                ohlcv_data =  await self.data_manager.collect_live_spot_ohlcv(base_symbol)
            elif instrument_type == 'future':
                ohlcv_data = await self.data_manager.collect_live_futures_ohlcv(base_symbol)
            else:
                raise ValueError(f"Invalid price type: {instrument_type}")
                
            if not ohlcv_data.empty:
                price = float(ohlcv_data.iloc[-1]['close'])
                self._price_cache[cache_key] = price
                return price
        except Exception as e:
            logger.error(f"Error getting current {instrument_type} price for {symbol}: {e}")
        
        return None
    
    async def pnl_close_position(self, symbol: str, instrument_type: str):
        """Update PnL for all perpetual positions"""
        current_price = await self.get_current_price(symbol, instrument_type)
        pos = self.positions[symbol]
        pnl = 0.0
        if current_price and pos['quantity'] != 0:
            if pos['side'] == 'LONG':
                pnl = pos['quantity'] * (current_price - pos['entry_price'])
            elif pos['side'] == 'SHORT':
                pnl = pos['quantity'] * (pos['entry_price'] - current_price)
            pos['value'] = abs(pos['quantity']) * current_price
        return pnl

    async def get_total_portfolio_value(self) -> float:
        """Get total portfolio value including balances and appropriate position valuation.

        - Spot: add current notional value of holdings
        - Futures: add unrealized PnL only (no principal counted)
        """
        total_value = self.balance.get('USDT', 0.0)

        for symbol, pos in self.positions.items():
            instrument_type = pos.get('instrument_type', 'future' if symbol.endswith('-PERP') else 'spot')
            current_price = await self.get_current_price(symbol, instrument_type)
            if not current_price:
                continue
            quantity = float(pos.get('quantity', 0.0))
            side = pos.get('side', 'LONG')
            entry = float(pos.get('entry_price', 0.0))

            if instrument_type == 'future':
                # Unrealized PnL contribution only
                if abs(quantity) > 0:
                    if side == 'LONG':
                        unrealized = quantity * (current_price - entry)
                    elif side == 'SHORT':
                        unrealized = quantity * (entry - current_price)
                    else:
                        unrealized = 0.0
                    total_value += unrealized
                # Maintain notional and latest value fields for reporting
                pos['value'] = abs(quantity) * current_price
            else:
                # Spot: full notional value contributes to portfolio value
                pos['value'] = abs(quantity) * current_price
                total_value += pos['value']
        return total_value

    async def get_position_summary(self) -> Dict[str, Any]:
        """Get a summary of all positions and balances"""
        summary = {
            'balances': self.balance.copy(),
            'positions': {},
            'total_portfolio_value': await self.get_total_portfolio_value(),
            'total_trades': len(self.trade_history)
        }
        
        for symbol, pos in self.positions.items():
            if abs(pos['quantity']) > 0:
                instrument_type = pos.get('instrument_type', 'future' if symbol.endswith('-PERP') else 'spot')
                current_price = await self.get_current_price(symbol, instrument_type)
                if current_price:
                    qty = float(pos['quantity'])
                    entry = float(pos['entry_price'])
                    if instrument_type == 'future':
                        unrealized = qty * ((current_price - entry) if pos['side'] == 'LONG' else (entry - current_price))
                        summary['positions'][symbol] = {
                            'quantity': qty,
                            'side': pos['side'],
                            'entry_price': entry,
                            'current_price': current_price,
                            'unrealized_pnl': unrealized,
                            'value': abs(qty) * current_price
                        }
                    else:
                        summary['positions'][symbol] = {
                            'quantity': qty,
                            'side': pos['side'],
                            'entry_price': entry,
                            'current_price': current_price,
                            'pnl': qty * (current_price - entry),
                            'value': abs(qty) * current_price
                        }
        
        return summary

