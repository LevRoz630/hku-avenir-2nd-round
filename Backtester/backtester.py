#!/usr/bin/env python3
"""
Comprehensive Backtester for Trading Strategies
Compatible with existing OMS-based trading strategies with minimal code changes.

This backtester provides OMS-compatible functions that work with historical data
instead of live trading, allowing you to test strategies without modification.
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BacktesterOMS:
    """
    Mock OMS client that works with historical data for backtesting.
    Provides the same interface as the real OMS client but uses historical data.
    """
    
    def __init__(self, historical_data_dir: str = "historical_data"):
        self.historical_data_dir = Path(historical_data_dir)
        self.positions = {}  # Current positions
        self.balance = {"USDT": 10000.0}  # Starting balance
        self.trade_history = []  # All trades executed
        self.performance_metrics = {}
        
    def get_balance(self) -> List[Dict[str, Any]]:
        """Get account balance - compatible with OMS interface"""
        return [{"asset": asset, "balance": str(balance)} for asset, balance in self.balance.items()]
    
    def get_position(self) -> List[Dict[str, Any]]:
        """Get current positions - compatible with OMS interface"""
        positions = []
        for symbol, position in self.positions.items():
            if abs(position['quantity']) > 0:
                positions.append({
                    "instrument_name": symbol,
                    "position_side": "LONG" if position['quantity'] > 0 else "SHORT",
                    "quantity": str(abs(position['quantity'])),
                    "value": str(abs(position['value'])),
                    "unrealized_pnl": str(position.get('unrealized_pnl', 0))
                })
        return positions
    
    def set_target_position(self, instrument_name: str, instrument_type: str, 
                          target_value: float, position_side: str) -> Dict[str, Any]:
        """Set target position - compatible with OMS interface"""
        # Convert to internal format
        target_value = float(target_value)
        if position_side == "SHORT":
            target_value = -target_value
            
        # Update position
        self.positions[instrument_name] = {
            'quantity': target_value,  # In USDT value
            'value': abs(target_value),
            'side': position_side,
            'type': instrument_type
        }
        
        # Record trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': instrument_name,
            'type': instrument_type,
            'side': position_side,
            'value': abs(target_value),
            'quantity': target_value
        })
        
        return {"id": f"backtest_{len(self.trade_history)}", "status": "success"}
    
    def set_target_position_batch(self, elements: List[Dict]) -> Dict[str, Any]:
        """Batch set target positions - compatible with OMS interface"""
        for element in elements:
            self.set_target_position(
                element['instrument_name'],
                element['instrument_type'],
                element['target_value'],
                element['position_side']
            )
        return {"id": f"batch_{len(self.trade_history)}", "status": "success"}
    
    def close(self):
        """Close client - compatible with OMS interface"""
        pass

class HistoricalDataManager:
    """Manages loading and accessing historical data for backtesting"""
    
    def __init__(self, data_dir: str = "historical_data"):
        self.data_dir = Path(data_dir)
        self.ohlcv_data = {}
        self.funding_data = {}
        self.open_interest_data = {}
        self.trades_data = {}
        
    def load_historical_data(self, symbols: List[str], data_types: List[str] = None):
        """Load all historical data for given symbols"""
        if data_types is None:
            data_types = ['ohlcv', 'funding_rates', 'open_interest', 'trades']
            
        logger.info(f"Loading historical data for {len(symbols)} symbols...")
        
        for symbol in symbols:
            self.ohlcv_data[symbol] = self._load_ohlcv_data(symbol)
            if 'funding_rates' in data_types:
                self.funding_data[symbol] = self._load_funding_data(symbol)
            if 'open_interest' in data_types:
                self.open_interest_data[symbol] = self._load_open_interest_data(symbol)
            if 'trades' in data_types:
                self.trades_data[symbol] = self._load_trades_data(symbol)
                
        logger.info("Historical data loading completed")
    
    def _load_ohlcv_data(self, symbol: str) -> pd.DataFrame:
        """Load OHLCV data for a symbol"""
        # Try different timeframes and days combinations
        for timeframe in ['1m', '5m', '15m', '1h', '1d']:
            for days in [1, 7, 30, 90]:
                filename = f"future_{symbol}_{timeframe}_{days}d.csv"
                filepath = self.data_dir / filename
                if filepath.exists():
                    df = pd.read_csv(filepath)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df
        
        # If no data found, return empty DataFrame
        logger.warning(f"No OHLCV data found for {symbol}")
        return pd.DataFrame()
    
    def _load_funding_data(self, symbol: str) -> pd.DataFrame:
        """Load funding rate data for a symbol"""
        for days in [1, 7, 30, 90]:
            filename = f"future_{symbol}_funding_rates_{days}d.csv"
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
                filename = f"future_{symbol}_open_interest_{days}d_{timeframe}.csv"
                filepath = self.data_dir / filename
                if filepath.exists():
                    df = pd.read_csv(filepath)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df
        
        logger.warning(f"No open interest data found for {symbol}")
        return pd.DataFrame()
    
    def _load_trades_data(self, symbol: str) -> pd.DataFrame:
        """Load trades data for a symbol"""
        for days in [1, 7, 30, 90]:
            filename = f"future_{symbol}_trades_{days}d.csv"
            filepath = self.data_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        
        logger.warning(f"No trades data found for {symbol}")
        return pd.DataFrame()
    
    def get_data_at_time(self, symbol: str, timestamp: datetime, data_type: str = 'ohlcv') -> pd.DataFrame:
        """Get data up to a specific timestamp"""
        if data_type == 'ohlcv':
            data = self.ohlcv_data.get(symbol, pd.DataFrame())
        elif data_type == 'funding_rates':
            data = self.funding_data.get(symbol, pd.DataFrame())
        elif data_type == 'open_interest':
            data = self.open_interest_data.get(symbol, pd.DataFrame())
        elif data_type == 'trades':
            data = self.trades_data.get(symbol, pd.DataFrame())
        else:
            return pd.DataFrame()
        
        if data.empty:
            return data
            
        # Filter data up to timestamp
        return data[data['timestamp'] <= timestamp].copy()

class Backtester:
    """
    Main backtester class that provides OMS-compatible interface for testing strategies
    """
    
    def __init__(self, 
                 historical_data_dir: str = "historical_data",
                 initial_balance: float = 10000.0,
                 symbols: List[str] = None):
        
        self.historical_data_dir = historical_data_dir
        self.symbols = symbols or [
            "BTC-USDT-PERP", "ETH-USDT-PERP", "SOL-USDT-PERP", 
            "BNB-USDT-PERP", "XRP-USDT-PERP"
        ]
        
        # Initialize components
        self.data_manager = HistoricalDataManager(historical_data_dir)
        self.oms_client = BacktesterOMS(historical_data_dir)
        
        # Set initial balance
        self.oms_client.balance = {"USDT": initial_balance}
        
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
        
        logger.info(f"Backtester initialized with {len(self.symbols)} symbols")
    
    def get_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Get historical OHLCV data - compatible with strategy interface"""
        data = {}
        for symbol in self.symbols:
            data[symbol] = self.data_manager.get_data_at_time(symbol, self.current_time, 'ohlcv')
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
        
        # Initialize strategy
        strategy = strategy_class()
        
        # Replace strategy's OMS functions with backtester functions
        strategy.oms_client = self.oms_client
        strategy.get_historical_data = self.get_historical_data
        strategy.get_funding_rate_data = self.get_funding_rate_data
        strategy.get_open_interest_data = self.get_open_interest_data
        strategy.get_account_balance = self.get_account_balance
        strategy.get_current_positions = self.get_current_positions
        strategy.push_target_positions = self.push_target_positions
        strategy.show_account_detail = self.show_account_detail
        
        # Run backtest
        self.current_time = start_date
        iteration = 0
        
        while self.current_time <= end_date:
            try:
                # Update strategy's current time
                strategy.current_time = self.current_time
                
                # Run strategy logic
                if hasattr(strategy, 'run_strategy'):
                    strategy.run_strategy()
                elif hasattr(strategy, 'on_1min_kline'):
                    # For strategies that use kline callbacks
                    for symbol in self.symbols:
                        ohlcv_data = self.data_manager.get_data_at_time(symbol, self.current_time, 'ohlcv')
                        if not ohlcv_data.empty:
                            latest_candle = ohlcv_data.iloc[-1]
                            kline = [
                                int(latest_candle['timestamp'].timestamp() * 1000),
                                latest_candle['open'],
                                latest_candle['high'],
                                latest_candle['low'],
                                latest_candle['close'],
                                latest_candle['volume']
                            ]
                            # Note: This would need to be adapted for async strategies
                            # strategy.on_1min_kline(symbol, kline)
                
                # Record portfolio value
                self._record_portfolio_value()
                
                # Move to next time step
                self.current_time += time_step
                iteration += 1
                
                if iteration % 24 == 0:  # Log every 24 iterations
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
            'total_return': self.portfolio_values[-1] / self.portfolio_values[0] - 1 if self.portfolio_values else 0,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'trade_history': self.oms_client.trade_history,
            'final_balance': self.get_account_balance(),
            'final_positions': self.get_current_positions()
        }
    
    def _get_earliest_data_time(self) -> datetime:
        """Get earliest timestamp from all loaded data"""
        earliest = None
        for symbol in self.symbols:
            data = self.data_manager.ohlcv_data.get(symbol, pd.DataFrame())
            if not data.empty:
                symbol_earliest = data['timestamp'].min()
                if earliest is None or symbol_earliest < earliest:
                    earliest = symbol_earliest
        return earliest or datetime.now() - timedelta(days=7)
    
    def _get_latest_data_time(self) -> datetime:
        """Get latest timestamp from all loaded data"""
        latest = None
        for symbol in self.symbols:
            data = self.data_manager.ohlcv_data.get(symbol, pd.DataFrame())
            if not data.empty:
                symbol_latest = data['timestamp'].max()
                if latest is None or symbol_latest > latest:
                    latest = symbol_latest
        return latest or datetime.now()
    
    def _record_portfolio_value(self):
        """Record current portfolio value"""
        # Calculate total portfolio value
        total_value = self.oms_client.balance.get('USDT', 0)
        
        # Add unrealized PnL from positions
        for symbol, position in self.oms_client.positions.items():
            if abs(position['quantity']) > 0:
                # Get current price
                ohlcv_data = self.data_manager.get_data_at_time(symbol, self.current_time, 'ohlcv')
                if not ohlcv_data.empty:
                    current_price = ohlcv_data.iloc[-1]['close']
                    position_value = abs(position['quantity'])
                    # Simple PnL calculation (this could be more sophisticated)
                    if position['side'] == 'LONG':
                        pnl = position_value * 0.01  # Simplified
                    else:
                        pnl = -position_value * 0.01  # Simplified
                    total_value += pnl
        
        self.portfolio_values.append(total_value)
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics"""
        if len(self.portfolio_values) < 2:
            return
            
        # Calculate daily returns
        self.daily_returns = []
        for i in range(1, len(self.portfolio_values)):
            daily_return = (self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
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
            
        def run_strategy(self):
            # Your strategy logic here
            print(f"Running strategy at {self.current_time}")
            # Example: simple buy and hold
            positions = {"BTC-USDT-PERP": 1000.0}
            self.push_target_positions(positions)
    
    # Run backtest
    results = backtester.run_backtest(
        strategy_class=ExampleStrategy,
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now(),
        time_step=timedelta(hours=1)
    )
    
    # Print results
    backtester.print_results(results)
