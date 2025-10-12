#!/usr/bin/env python3
"""
Long BTC / Short Altcoin Portfolio Strategy (v3)

Strategy Logic:
- Long Bitcoin futures as a hedge
- Short altcoins with negative 24h returns
- Position sizing based on variance-adjusted weights
- Rebalance when BTC/Alt ratio changes or daily for altcoin weights

Adapted for backtest_local framework with PositionManager integration.
Returns orders instead of directly controlling OMS.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from hist_data import HistoricalDataCollector
from oms_simulation import OMSClient


class BTCAltShortStrategy:
    def __init__(self, symbols: List[str], historical_data_dir: str = None, lookback_days: int = 5, 
                 current_btc_ratio: float = 0.3, drift_threshold: float = 0.20):
        """
        Initialize BTC long / Altcoin short strategy.
        
        Args:
            symbols: List of symbols including BTC-USDT and altcoins
            historical_data_dir: Path to historical data (unused in backtest_local - kept for compatibility)
            lookback_days: Days of history for variance calculation
            current_btc_ratio: Target BTC allocation ratio (e.g., 0.3 = 30% BTC, 70% alts)
            drift_threshold: Rebalance trigger when ratio drifts by this amount (e.g., 0.20 = 20%)
        """
        self.symbols = symbols
        self.historical_data_dir = historical_data_dir
        
        # Identify BTC and altcoins
        self.btc_symbol = None
        self.altcoin_symbols = []
        
        for sym in symbols:
            perp_symbol = sym + '-PERP' if not sym.endswith('-PERP') else sym
            base = perp_symbol.replace('-USDT', '').replace('-PERP', '')
            if base == 'BTC':
                self.btc_symbol = perp_symbol
            else:
                self.altcoin_symbols.append(perp_symbol)
        if not self.btc_symbol:
            raise ValueError("BTC-USDT must be included in symbols")
        if not self.altcoin_symbols:
            raise ValueError("At least one altcoin must be included")
        
        # State tracking
        self.last_rebalance_day = None
        self.start_time = None
        self.end_time = None
        
        # Strategy parameters
        self.lookback_days = lookback_days
        self.current_btc_ratio = current_btc_ratio  
        self.drift_threshold = drift_threshold

        
    def _get_hourly_data(self, base_symbol: str, hours: int, oms_client: OMSClient, 
                         data_manager: HistoricalDataCollector) -> pd.DataFrame:
        """Get hourly index price data for a symbol."""
        dm = data_manager
        
        window_start = oms_client.current_time - pd.Timedelta(hours=hours)
        end_time = oms_client.current_time
        
        df = dm.perpetual_index_ohlcv_data.get(base_symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df[df['timestamp'].between(window_start, end_time, inclusive='left')]
        df = df.sort_values('timestamp')
        
        if df.empty or len(df) < 24:
            return 0.0
        
        # Get price 24 hours ago and current price
        df = df.set_index('timestamp')
        hourly = df['close'].resample('1H').last().dropna()
        

        
        return hourly
    
    def _calculate_log_returns(self, base_symbol: str, oms_client: OMSClient, 
                               data_manager: HistoricalDataCollector) -> float:
        """Calculate 24h log returns for a symbol."""
        hourly = self._get_hourly_data(base_symbol, hours=48, oms_client=oms_client, data_manager=data_manager)
        
        if len(hourly) < 24:
            return 0.0

        price_24h_ago = hourly.iloc[-25] if len(hourly) >= 25 else hourly.iloc[0]
        current_price = hourly.iloc[-1]
        
        log_return = np.log(current_price / price_24h_ago)
        return float(log_return)
    
    def _calculate_variance(self, base_symbol: str, oms_client: OMSClient, 
                            data_manager: HistoricalDataCollector) -> float:
        """Calculate variance of returns over lookback period."""
        hourly = self._get_hourly_data(base_symbol, hours=self.lookback_days * 24, 
                                    oms_client=oms_client, data_manager=data_manager)
        
        if len(hourly) < 48:
            return 1.0
        
        # Calculate 24h rolling returns
        returns = np.log(hourly / hourly.shift(24)).dropna()
        
        if len(returns) < 2:
            return 1.0
        
        variance = float(returns.var())
        return max(variance, 1e-8)  
    
    def _calculate_altcoin_weights(self, oms_client: OMSClient, 
                                    data_manager: HistoricalDataCollector) -> Dict[str, float]:
        """
        Calculate altcoin portfolio weights based on strategy rules.
        
        Returns:
            Dict mapping symbol to weight (0-1), only including assets >10%
        """
        weights = {}
        
        for symbol in self.altcoin_symbols:
            base = symbol.replace('-PERP', '')
            
            # Get 24h log returns
            log_return = self._calculate_log_returns(base, oms_client, data_manager)
            
            # Signal: 1 if negative return, 0 otherwise
            signal = 1.0 if log_return < 0 else 0.0
            
            if signal == 0:
                continue  
            
            # Get variance
            variance = self._calculate_variance(base, oms_client, data_manager)
            
            # Raw weight: signal * mod(log_return) / variance
            raw_weight = signal * abs(log_return) / variance
            weights[symbol] = raw_weight
        
        if not weights:
            return {}
        
        # Normalize to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Filter: only keep weights > 10% we want to avoid fees when entering and exiting too much positions
        weights = {k: v for k, v in weights.items() if v > 0.10}
        
        # Renormalize after filtering
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _should_rebalance_ratio(self, oms_client: OMSClient) -> bool:
        """Check if BTC/Alt ratio has drifted (drift_threshold) from target."""
        positions = oms_client.get_position()
        
        drift_threshold = self.drift_threshold

        btc_value = 0.0
        alt_value = 0.0
        
        for pos in positions:
            symbol = pos['symbol']
            value = abs(float(pos.get('value', 0.0)))
            
            if symbol == self.btc_symbol:
                btc_value = value
            elif symbol in self.altcoin_symbols:
                alt_value += value
        
        total_value = btc_value + alt_value
        if total_value < 1.0:
            return True  # No positions, need to establish
        
        current_btc_ratio = btc_value / total_value
        
        drift = abs(current_btc_ratio - self.current_btc_ratio)
        
        return drift > drift_threshold
    
    def run_strategy(self, oms_client: OMSClient, data_manager: HistoricalDataCollector) -> List[Dict[str, Any]]:
        """
        Execute strategy logic and return orders for PositionManager to process.
        
        Args:
            oms_client: OMS client for portfolio state
            data_manager: Historical data collector for price data
            
        Returns:
            List of order dicts with keys: symbol, instrument_type, side, value (optional)
            PositionManager will size the orders based on risk parameters.
        """
        now = oms_client.current_time
        total_equity = float(oms_client.get_total_portfolio_value() or 10000.0)
        
        # Check if we should rebalance (once per day)
        current_day = (now.year, now.month, now.day)
        should_rebalance_daily = self.last_rebalance_day != current_day
        should_rebalance_ratio = self._should_rebalance_ratio(oms_client)
        
        if not (should_rebalance_daily or should_rebalance_ratio):
            return []  # No action needed - return empty orders
        
        print(f"\n{'='*60}")
        print(f"REBALANCING at {now}")
        print(f"Reason: {'Daily' if should_rebalance_daily else f'Ratio drift >{self.drift_threshold:.0%}'}")
        print(f"{'='*60}")
        
        # Calculate altcoin weights
        altcoin_weights = self._calculate_altcoin_weights(oms_client, data_manager)
        
        # Build orders list to return to PositionManager
        orders: List[Dict[str, Any]] = []
        
        if not altcoin_weights:
            print("No altcoins with negative returns. Closing all positions.")
            # Close all positions
            orders.append({'symbol': self.btc_symbol, 'instrument_type': 'future', 'side': 'CLOSE'})
            for alt_sym in self.altcoin_symbols:
                orders.append({'symbol': alt_sym, 'instrument_type': 'future', 'side': 'CLOSE'})
            return orders
        
        print(f"\nAltcoin weights (filtered >10%):")
        for sym, weight in altcoin_weights.items():
            base = sym.replace('-PERP', '')
            log_ret = self._calculate_log_returns(base, oms_client, data_manager)
            print(f"  {sym}: {weight:.2%} (24h log return: {log_ret:.4f})")
        
        # Allocate capital: use leverage carefully
        # Total allocation = 90% of equity (keep 10% buffer)
        total_allocation = 0.9 * total_equity
        
        # BTC and alt allocations based on target ratio
        btc_allocation = self.current_btc_ratio * total_allocation
        alt_allocation = (1.0 - self.current_btc_ratio) * total_allocation
        
        print(f"\nAllocation:")
        print(f"  Total equity: ${total_equity:.2f}")
        print(f"  BTC long: ${btc_allocation:.2f} ({self.current_btc_ratio:.1%})")
        print(f"  Alt short: ${alt_allocation:.2f} ({1.0 - self.current_btc_ratio:.1%})")
        
        # Build BTC long order
        orders.append({
            'symbol': self.btc_symbol,
            'instrument_type': 'future',
            'side': 'LONG',
            'value': btc_allocation  # Pre-sized by strategy; PositionManager can adjust if needed
        })
        
        # Build altcoin short orders
        for alt_sym, weight in altcoin_weights.items():
            alt_usdt = alt_allocation * weight
            orders.append({
                'symbol': alt_sym,
                'instrument_type': 'future',
                'side': 'SHORT',
                'value': alt_usdt  # Pre-sized by strategy
            })
        
        # Close positions for altcoins not in current portfolio
        current_positions = oms_client.get_position()
        for pos in current_positions:
            symbol = pos['symbol']
            if symbol in self.altcoin_symbols and symbol not in altcoin_weights:
                orders.append({'symbol': symbol, 'instrument_type': 'future', 'side': 'CLOSE'})
        
        # Update state
        self.last_rebalance_day = current_day
        
        print(f"\nRebalance complete at {now}")
        print(f"Generated {len(orders)} orders for PositionManager")
        print(f"{'='*60}\n")
        
        return orders