#!/usr/bin/env python3
"""
Long BTC / Short Altcoin Portfolio Strategy

Strategy Logic:
- Long Bitcoin futures as a hedge
- Short altcoins with negative 24h returns
- Position sizing based on variance-adjusted weights
- Rebalance when BTC/Alt ratio changes >20% or daily for altcoin weights
"""

from typing import List, Dict
import numpy as np
import pandas as pd
from oms_simulation import OMSClient
from hist_data import HistoricalDataCollector


class BTCAltShortStrategy:
    def __init__(self, symbols: List[str], historical_data_dir: str, lookback_days: int = 30, current_btc_ratio: float = 0.3, drift_threshold: float = 0.20):
        """
        Initialize BTC long / Altcoin short strategy.
        
        Args:
            symbols: List of symbols including BTC-USDT and altcoins
            historical_data_dir: Path to historical data
            lookback_days: Days of history for variance calculation
        """
        self.symbols = symbols
        self.oms_client = None 
        self.data_manager = None

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
        
        self.last_rebalance_day = None
        self.current_time = None
        self.end_time = None
        

        # Params
        self.lookback_days = lookback_days
        self.current_btc_ratio = current_btc_ratio  
        self.drift_threshold = drift_threshold

        
    def _get_hourly_data(self, base_symbol: str, hours: int = 48) -> pd.DataFrame:
        """Get hourly index price data for a symbol."""
        dm = self.oms_client.data_manager
        
        window_start = self.oms_client.current_time - pd.Timedelta(hours=hours)
        end_time = self.oms_client.current_time
        
        df = dm.perpetual_index_ohlcv_data.get(base_symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df[df['timestamp'].between(window_start, end_time, inclusive='left')]
        df = df.sort_values('timestamp')
        
        return df
    
    def _calculate_log_returns(self, base_symbol: str) -> float:
        """Calculate 24h log returns for a symbol."""
        df = self._get_hourly_data(base_symbol, hours=48)
        
        if df.empty or len(df) < 24:
            return 0.0
        
        # Get price 24 hours ago and current price
        df = df.set_index('timestamp')
        hourly = df['close'].resample('1h').last().dropna()
        
        if len(hourly) < 24:
            return 0.0
        
        price_24h_ago = hourly.iloc[-25] if len(hourly) >= 25 else hourly.iloc[0]
        current_price = hourly.iloc[-1]
        
        log_return = np.log(current_price / price_24h_ago)
        return float(log_return)
    
    def _calculate_variance(self, base_symbol: str) -> float:
        """Calculate variance of returns over lookback period."""
        df = self._get_hourly_data(base_symbol, hours=self.lookback_days * 24)
        
        if df.empty or len(df) < 48:
            return 1.0  # Default variance
        
        df = df.set_index('timestamp')
        hourly = df['close'].resample('1H').last().dropna()
        
        if len(hourly) < 48:
            return 1.0
        
        # Calculate 24h rolling returns
        returns = np.log(hourly / hourly.shift(24)).dropna()
        
        if len(returns) < 2:
            return 1.0
        
        variance = float(returns.var())
        return max(variance, 1e-8)  
    
    def _calculate_altcoin_weights(self) -> Dict[str, float]:
        """
        Calculate altcoin portfolio weights based on strategy rules.
        
        Returns:
            Dict mapping symbol to weight (0-1), only including assets >10%
        """
        weights = {}
        
        for symbol in self.altcoin_symbols:
            base = symbol.replace('-PERP', '')
            
            # Get 24h log returns
            log_return = self._calculate_log_returns(base)
            
            # Signal: 1 if negative return, 0 otherwise
            signal = 1.0 if log_return < 0 else 0.0
            
            if signal == 0:
                continue  
            
            # Get variance
            variance = self._calculate_variance(base)
            
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
    
    def _should_rebalance_ratio(self) -> bool:
        """Check if BTC/Alt ratio has drifted (drift_threshold) from target."""
        positions = self.oms_client.get_position()
        
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
    
    def run_strategy(self, oms_client: OMSClient, data_manager: HistoricalDataCollector):
        """Execute strategy logic."""
        orders = []
        self.oms_client = oms_client
        self.data_manager = data_manager
        now = self.oms_client.current_time
        
        # Check if we should rebalance (once per day)
        current_day = (now.year, now.month, now.day)
        should_rebalance_daily = self.last_rebalance_day != current_day
        should_rebalance_ratio = self._should_rebalance_ratio()
        
        if not (should_rebalance_daily or should_rebalance_ratio):
            return []  # No action needed
        
        print(f"\n{'='*60}")
        print(f"REBALANCING at {now}")
        print(f"Reason: {'Daily' if should_rebalance_daily else 'Ratio drift >20%'}")
        print(f"{'='*60}")
        
        # Calculate altcoin weights
        altcoin_weights = self._calculate_altcoin_weights()
        
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
            log_ret = self._calculate_log_returns(base)
            print(f"  {sym}: {weight:.2%} (24h log return: {log_ret:.4f})")
        
        print(f"\nAllocation:")
        
        print(f"  BTC long: ${self.current_btc_ratio:.2f}")
        print(f"  Alt short: ${1.0 - self.current_btc_ratio:.2f}")
        
        # adds the position with the alloc_frac that is later used by position manager to set the value
        orders.append({
            'symbol': self.btc_symbol, 
            'instrument_type': 'future', 
            'alloc_frac': self.current_btc_ratio, 
            'side': 'LONG'
        })
        
        # adds the position with the alloc_frac that is later used by position manager to set the value
        for alt_sym, weight in altcoin_weights.items():

            orders.append({
                'symbol': alt_sym, 
                'instrument_type': 'future', 
                'alloc_frac': (1.0 - self.current_btc_ratio) * weight, 
                'side': 'SHORT'
            })
        
        # closes the positions for altcoins not in the current portfolio
        for alt_sym in self.altcoin_symbols:
            if alt_sym not in altcoin_weights:
                orders.append({'symbol': alt_sym, 'instrument_type': 'future', 'side': 'CLOSE'})
        
        # updates the state
        self.last_rebalance_day = current_day
        
        print(f"\nRebalance complete at {now}")
        print(f"{'='*60}\n")
        return orders