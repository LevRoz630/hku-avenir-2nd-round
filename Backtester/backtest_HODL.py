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

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from backtester import Backtester
from strategy_adapter import run_backtest_with_strategy
from hist_data import HistoricalDataCollector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collect_historical_data():
    """Collect historical data for backtesting"""
    print("Collecting historical data...")
    
    # Define symbols to collect
    symbols = [
        "BTC-USDT",
        "ETH-USDT", 
        "SOL-USDT",
    ]

    # Initialize data collector
    collector = HistoricalDataCollector(
        data_dir="historical_data",
        days=30,  # Collect 30 days of data
        symbols=symbols
    )


    
    # Collect different types of data
    collector.collect_comprehensive_data(timeframe='1h')
    print("Historical data collection completed!")


def run_demo_strategy_backtest():
    """Run backtest with the demo strategy"""
    print("\n" + "="*60)
    print("RUNNING DEMO STRATEGY BACKTEST")
    print("="*60)
    
    # Initialize backtester
    backtester = Backtester(
        historical_data_dir="historical_data",
        initial_balance=10000.0,
        symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    )
    
    # Run backtest
    results = run_backtest_with_strategy(
        backtester,
        strategy_name="HODL",
        start_date=datetime.now() - timedelta(days=1),
        end_date=datetime.now(),
        time_step=timedelta(hours=1)
    )
    
    # Print results
    backtester.print_results(results)
    
    return results

def plot_performance(demo_results):
    """Plot performance comparison"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(12, 8))
        
        # Plot portfolio values
        plt.subplot(2, 2, 1)
        plt.plot(demo_results['portfolio_values'], label='Demo Strategy', alpha=0.8)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value (USDT)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        

        # Plot drawdown
        plt.subplot(2, 2, 3)
        demo_drawdown = []
        funding_drawdown = []
        
        # Calculate running drawdown
        demo_peak = demo_results['portfolio_values'][0]

        for value in demo_results['portfolio_values']:
            if value > demo_peak:
                demo_peak = value
            demo_drawdown.append((demo_peak - value) / demo_peak)
        
        plt.plot(demo_drawdown, label='Demo Strategy', alpha=0.8)
        plt.plot(funding_drawdown, label='Funding Rate Strategy', alpha=0.8)
        plt.title('Drawdown Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot performance metrics
        plt.subplot(2, 2, 4)
        metrics = ['Total Return', 'Max Drawdown', 'Sharpe Ratio']
        demo_values = [
            demo_results['total_return'],
            -demo_results['max_drawdown'],  # Negative for better visualization
            demo_results['sharpe_ratio']
        ]

        x = range(len(metrics))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], demo_values, width, label='Demo Strategy', alpha=0.8)
      
        
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
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Total trades: {len(trades)}")
    print(f"Trading frequency: {len(trades) / 7:.1f} trades per day")
    
    # Analyze by symbol
    symbol_counts = df['symbol'].value_counts()
    print("\nTrades by symbol:")
    for symbol, count in symbol_counts.items():
        print(f"  {symbol}: {count} trades")
    
    # Analyze by side
    side_counts = df['side'].value_counts()
    print("\nTrades by side:")
    for side, count in side_counts.items():
        print(f"  {side}: {count} trades")

def main():
    """Main function to run complete backtest analysis"""
    print("Crypto Trading Strategy Backtester")
    print("=" * 50)
    
    # Check if historical data exists
    data_dir = Path("historical_data")
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("No historical data found. Collecting data...")
        collect_historical_data()
    else:
        print("Historical data found. Skipping data collection.")
    
    # Run backtests
    demo_results = run_demo_strategy_backtest()

    # Analyze trades
    analyze_trades(demo_results, "Demo Strategy")
    
    # Plot results
    # plot_performance(demo_results)
    
    # Save results
    results_summary = {
        'demo_strategy': demo_results}
        
    import json
    from datetime import datetime
    
    with open('backtest_results.json', 'w') as f:
        # Convert various types for JSON serialization
        def convert_for_json(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                # Convert objects to dict, excluding circular references
                try:
                    return {k: convert_for_json(v) for k, v in obj.__dict__.items() 
                           if not k.startswith('_')}
                except:
                    return str(obj)
            else:
                return obj
        
        # Clean the results to avoid circular references
        clean_results = convert_for_json(results_summary)
        json.dump(clean_results, f, indent=2)
    
    print("\nBacktest results saved to 'backtest_results.json'")
    print("Backtest analysis completed!")

if __name__ == "__main__":
    main()
