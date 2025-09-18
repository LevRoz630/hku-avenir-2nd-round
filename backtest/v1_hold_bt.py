#!/usr/bin/env python3
"""
Backtest Example - How to use the backtester with existing trading strategies

This script demonstrates how to:
1. Collect historical data
2. Run backtests with existing strategies
3. Compare different strategies
4. Analyze results
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import logging

# Add current directory to path (for local strategies package)
sys.path.append(str(Path(__file__).parent))
# Add src directory to path (for core engine modules)
sys.path.append(str(Path(__file__).parent.parent / "src"))

from backtester import Backtester
from strategies.v1_hold import HoldStrategy
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_performance(demo_results):
    """Plot performance comparison"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(12, 8))
        

        plt.subplot(2, 2, 1)
        plt.plot(demo_results['portfolio_values'], label='Hold Strategy', alpha=0.8)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value (USDT)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        demo_drawdown = []
        funding_drawdown = []
        
        demo_peak = demo_results['portfolio_values'][0]

        for value in demo_results['portfolio_values']:
            if value > demo_peak:
                demo_peak = value
            demo_drawdown.append((demo_peak - value) / demo_peak)
        
        plt.plot(demo_drawdown, label='Hold Strategy', alpha=0.8)
        plt.plot(funding_drawdown, label='Funding Rate Strategy', alpha=0.8)
        plt.title('Drawdown Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        metrics = ['Total Return', 'Max Drawdown', 'Sharpe Ratio']
        demo_values = [
            demo_results['total_return'],
            -demo_results['max_drawdown'],  
            demo_results['sharpe_ratio']
        ]

        x = range(len(metrics))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], demo_values, width, label='Hold Strategy', alpha=0.8)
      
        
        plt.title('Performance Metrics Comparison')
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Performance chart saved as 'backtest_results.png'")
        
    except ImportError:
        print("Matplotlib not available. Skipping plotting.")
    except Exception as e:
        print(f"Error creating plot: {e}")

def analyze_trades(results, strategy_name):
    """Analyze trade patterns"""
    print(f"\n{strategy_name} Trade Analysis:")
    print("-" * 40)
    
    trades = results['trade_history']
    if not trades:
        print("No trades executed")
        return
    
    
    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Total trades: {len(trades)}")
    print(f"Trading frequency: {len(trades) / 7:.1f} trades per day")
    

    symbol_counts = df['symbol'].value_counts()
    print("\nTrades by symbol:")
    for symbol, count in symbol_counts.items():
        print(f"  {symbol}: {count} trades")

    side_counts = df['side'].value_counts()
    print("\nTrades by side:")
    for side, count in side_counts.items():
        print(f"  {side}: {count} trades")

def main():
    """Main function to run complete backtest analysis"""
    print("Crypto Trading Strategy Backtester")
    print("=" * 50)
    
    backtester = Backtester(
        historical_data_dir="historical_data",
    )
    
    # 4. Run backtest and change strategies and start date/end date and timestep

    results = backtester.run_backtest(strategy_class=HoldStrategy, 
    symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
    start_date=datetime.now() - timedelta(days = 15), 
    end_date=datetime.now(), time_step=timedelta(hours=1))

    
    backtester.print_results(results)

    # # Analyze trades
    # analyze_trades(results, "Hold Strategy")
    
    # Plot results
    plot_performance(results)
    
    # Save results
    results_summary = {
        'hold_strategy': results}

    data_dir = Path("historical_data")


if __name__ == "__main__":
    main()
