#!/usr/bin/env python3
"""
Strategy Adapter for Backtester
Makes existing OMS-based trading strategies compatible with the backtester
with minimal code changes.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add the parent directory to path to import strategies
parent_dir = Path(__file__).parent.parent / "avenir-hku-contest-demo-main"
sys.path.append(str(parent_dir))

logger = logging.getLogger(__name__)

class StrategyAdapter:
    """
    Adapter class that makes existing strategies compatible with the backtester.
    This class wraps your existing strategy and replaces OMS calls with backtester calls.
    """
    
    def __init__(self, strategy_class, backtester):
        """
        Initialize adapter with strategy class and backtester instance
        
        Args:
            strategy_class: Your existing strategy class (e.g., CryptoQuantDemo)
            backtester: Backtester instance
        """
        self.strategy_class = strategy_class
        self.backtester = backtester
        self.strategy = None
        
    def create_adapted_strategy(self):
        """Create an adapted strategy instance"""
        # Create strategy instance
        self.strategy = self.strategy_class()
        
        # Replace OMS-dependent methods with backtester methods
        self.strategy.oms_client = self.backtester.oms_client
        self.strategy.get_historical_data = self.backtester.get_historical_data
        self.strategy.get_funding_rate_data = self.backtester.get_funding_rate_data
        self.strategy.get_open_interest_data = self.backtester.get_open_interest_data
        self.strategy.get_account_balance = self.backtester.get_account_balance
        self.strategy.get_current_positions = self.backtester.get_current_positions
        self.strategy.push_target_positions = self.backtester.push_target_positions
        self.strategy.show_account_detail = self.backtester.show_account_detail
        
        # Set symbols from backtester
        self.strategy.symbols = self.backtester.symbols
        
        return self.strategy
    
    def run_strategy_at_time(self, current_time):
        """Run strategy at a specific time"""
        if not self.strategy:
            self.create_adapted_strategy()
        
        # Set current time
        self.strategy.current_time = current_time
        
        # Run strategy
        if hasattr(self.strategy, 'run_strategy'):
            self.strategy.run_strategy()
        else:
            logger.warning("Strategy does not have run_strategy method")

def adapt_demo_strategy(backtester):
    """
    Adapt the demo strategy from demo.py for backtesting
    
    Args:
        backtester: Backtester instance
        
    Returns:
        Adapted strategy instance
    """
    try:
        from demo import CryptoQuantDemo
        
        # Create adapter
        adapter = StrategyAdapter(CryptoQuantDemo, backtester)
        strategy = adapter.create_adapted_strategy()
        
        # Override the run_strategy method to work with backtester
        def adapted_run_strategy():
            """Adapted run_strategy that works with backtester"""
            print(f"\n===== Running Strategy at {strategy.current_time} =====")
            
            # Calculate 7-day returns using historical data
            returns = strategy.calculate_7d_returns()
            
            # Sort by returns
            sorted_returns = sorted(returns.items(), key=lambda x: x[1], reverse=True)
            
            # Select long and short positions
            long_symbols = [s for s, _ in sorted_returns[:strategy.position_switch_count]]
            short_symbols = [s for s, _ in sorted_returns[-strategy.position_switch_count:]]
            
            print("Long symbols:", long_symbols)
            print("Short symbols:", short_symbols)
            
            # Calculate position values
            total_usdt = strategy.get_account_balance()['USDT'] * 0.9
            position_value = total_usdt / (len(long_symbols) + len(short_symbols))
            
            # Build target positions
            target_positions = {}
            
            # Long positions (positive)
            for symbol in long_symbols:
                target_positions[symbol] = position_value
            
            # Short positions (negative)
            for symbol in short_symbols:
                target_positions[symbol] = -position_value
            
            # Push positions
            strategy.push_target_positions(target_positions)
        
        # Replace the run_strategy method
        strategy.run_strategy = adapted_run_strategy
        
        return strategy
        
    except ImportError as e:
        logger.error(f"Could not import demo strategy: {e}")
        return None

# def adapt_funding_rate_strategy(backtester):
#     """
#     Adapt the funding rate strategy from fundingrate2.py for backtesting
    
#     Args:
#         backtester: Backtester instance
        
#     Returns:
#         Adapted strategy instance
#     """
#     try:
#         from fundingrate2 import CryptoQuantDemo
        
#         # Create adapter
#         adapter = StrategyAdapter(CryptoQuantDemo, backtester)
#         strategy = adapter.create_adapted_strategy()
        
#         # Override the run_strategy method to work with backtester
#         def adapted_run_strategy():
#             """Adapted run_strategy that works with backtester"""
#             print(f"\n===== Running Funding Rate Strategy at {strategy.current_time} =====")
            
#             # Get funding rate data
#             funding_rate_data = strategy.get_funding_rate_data(periods=10)
#             funding_rate_ma = strategy.calculate_funding_rate_MA(periods=10)
#             funding_rate_signal = strategy.calculate_funding_rate_signal(periods=10)
            
#             # Calculate target positions
#             target_futures_positions = {}
#             target_spot_positions = {}
            
#             total_usdt = strategy.get_account_balance()['USDT'] * 0.3
#             position_value = total_usdt / len(strategy.symbols)
            
#             for symbol, signal_data in funding_rate_signal.items():
#                 if signal_data['signal']:
#                     target_futures_positions[symbol] = signal_data['direction'] * position_value
#                     # Hedging
#                     target_spot_positions[symbol] = -signal_data['direction'] * position_value
            
#             # Push positions
#             strategy.push_target_positions(target_futures_positions, type="future")
#             strategy.push_target_positions(target_spot_positions, type="spot")
        
#         # Replace the run_strategy method
#         strategy.run_strategy = adapted_run_strategy
        
#         return strategy
        
#     except ImportError as e:
#         logger.error(f"Could not import funding rate strategy: {e}")
#         return None

def HODL_strategy(backtester):
        # Simple HODL (buy and hold) strategy for backtester
        class HODLStrategy:
            def __init__(self):
                self.oms_client = None
                self.current_time = None
                self.has_bought = False

            def run_strategy(self):
                # Only buy once at the beginning, then hold
                if not self.has_bought:
                    # Allocate all USDT equally to all symbols (assume all are spot or perpetual)
                    balance = self.oms_client.get_account_balance()
                    usdt = balance.get("USDT", 0)
                    if usdt > 0:
                        num_symbols = len(self.symbols)
                        if num_symbols == 0:
                            return
                        allocation = (usdt - 1000) / num_symbols
                        for symbol in self.symbols:
                            # Determine instrument type
                            self.oms_client.set_target_position(
                                instrument_name=symbol,
                                instrument_type="spot",
                                target_value=allocation,
                                position_side="LONG"
                            )
                            self.oms_client.set_target_position(
                                instrument_name=symbol,
                                instrument_type="future",
                                target_value=10,
                                position_side="LONG"
                            )
                        self.has_bought = True

        # Attach symbols from backtester if available
        strategy = HODLStrategy()
        if hasattr(backtester, "symbols"):
            strategy.symbols = backtester.symbols
        strategy.oms_client = backtester.oms_client
        strategy.current_time = getattr(backtester, "current_time", None)
        
        return strategy
        


def run_backtest_with_strategy(backtester, strategy_name="demo", **kwargs):
    """
    Run backtest with a specific strategy
    
    Args:
        backtester: Backtester instance
        strategy_name: Name of strategy ("demo" or "funding_rate")
        **kwargs: Additional arguments for backtest
        
    Returns:
        Backtest results
    """
    
    if strategy_name == "demo":
        strategy = adapt_demo_strategy(backtester)
    # elif strategy_name == "funding_rate":
    #     strategy = adapt_funding_rate_strategy(backtester)
    elif strategy_name == "HODL":
        strategy = HODL_strategy(backtester)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    if not strategy:
        raise RuntimeError(f"Could not create strategy: {strategy_name}")
    
    # Run backtest
    results = backtester.run_backtest(
        strategy_class=lambda: strategy,
        **kwargs
    )
    
    return results

# Example usage
if __name__ == "__main__":
    from backtester import Backtester
    from datetime import datetime, timedelta
    
    # Initialize backtester
    backtester = Backtester(
        historical_data_dir="historical_data",
        initial_balance=10000.0,
        symbols=["BTC-USDT-PERP", "ETH-USDT-PERP", "SOL-USDT-PERP"]
    )
    
    # Run backtest with demo strategy
    print("Running backtest with demo strategy...")
    results = run_backtest_with_strategy(
        backtester,
        strategy_name="demo",
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now(),
        time_step=timedelta(hours=1)
    )
    
    # Print results
    backtester.print_results(results)
    
# def adapt_funding_rate_strategy2(backtester):
#     try:
#         from fundingrate2 import CryptoQuantDemo
        
#         # Create adapter
#         adapter = StrategyAdapter(CryptoQuantDemo, backtester)
#         strategy = adapter.create_adapted_strategy()
        
#         # Override the run_strategy method to work with backtester
#         def adapted_run_strategy():
#             """Adapted run_strategy that works with backtester"""
#             print(f"\n===== Running Funding Rate Strategy at {strategy.current_time} =====")
            
#             # Get funding rate data
#             funding_rate_data = strategy.get_funding_rate_data(periods=10)
#             funding_rate_ma = strategy.calculate_funding_rate_MA(periods=10)
#             funding_rate_signal = strategy.calculate_funding_rate_signal(periods=10)
            
#             # Calculate target positions
#             target_futures_positions = {}
#             target_spot_positions = {}
            
#             total_usdt = strategy.get_account_balance()['USDT'] * 0.3
#             position_value = total_usdt / len(strategy.symbols)
            
#             for symbol, signal_data in funding_rate_signal.items():
#                 if signal_data['signal']:
#                     target_futures_positions[symbol] = signal_data['direction'] * position_value
#                     # Hedging
#                     target_spot_positions[symbol] = -signal_data['direction'] * position_value
            
#             # Push positions
#             strategy.push_target_positions(target_futures_positions, type="future")
#             strategy.push_target_positions(target_spot_positions, type="spot")
        
#         # Replace the run_strategy method
#         strategy.run_strategy = adapted_run_strategy
        
#         return strategy
        
#     except ImportError as e:
#         logger.error(f"Could not import funding rate strategy: {e}")
#         return None
