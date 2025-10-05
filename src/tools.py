from datetime import datetime, timezone
import pandas as pd

# Helper methods with documentation
def _convert_symbol_to_ccxt(symbol: str, market_type: str = "spot"):
    """
    Convert trading symbol format for CCXT API calls.
    
    Parameters
    ----------
    symbol : str
        Trading symbol in standard format (e.g., "BTC-USDT", "ETH-USDT-PERP")
    market_type : str, optional
        Market type: "spot" or "future". Default is "spot".
    
    Returns
    -------
    str or None
        Converted symbol format for CCXT API calls.
        For spot: "BTC-USDT" -> "BTC/USDT"
        For futures: "BTC-USDT-PERP" -> "BTC/USDT:USDT"
        Returns None if conversion fails.
    """
    try:
        if market_type == "spot":
            # For spot: BTC-USDT -> BTC/USDT
            if '-' in symbol:
                return symbol.replace('-', '/')
            return symbol
        else:
            # For futures: BTC-USDT-PERP -> BTC/USDT:USDT
            base = symbol.split('-')[0]
            return f"{base}/USDT:USDT"
    except:
        return None

def _normalize_symbol_pair(symbol: str) -> str | None:
    """
    Normalize various symbol inputs to CCXT pair format BASE/QUOTE.
    Examples:
        - "BTC-USDT-PERP" -> "BTC/USDT"
        - "BTC-USDT" -> "BTC/USDT"
        - "BTC/USDT" -> "BTC/USDT" (unchanged)
    """
    try:
        s = str(symbol).strip()
        if not s:
            return None
        # Already in CCXT pair format
        if '/' in s:
            base, quote = s.split('/', 1)
            return f"{base.upper()}/{quote.upper()}"
        # Remove potential "-PERP" suffix and split by '-'
        s = s.upper().replace("-PERP", "")
        parts = [p for p in s.split('-') if p]
        if len(parts) == 1:
            base = parts[0]
            quote = 'USDT'
        else:
            base, quote = parts[0], parts[1]
        return f"{base}/{quote}"
    except Exception:
        return None
            

def _is_utc(dt: datetime) -> None:
    if dt.tzinfo is timezone.utc:
        return
    else:
        raise ValueError("Timestamp is not in UTC timezone, please use datetime.now(timezone.utc) or similar")


def _get_number_of_periods(timeframe: str, start_time: datetime, end_time: datetime):
    """
    Calculate the number of periods between two datetime objects for a given timeframe.
    
    Parameters
    ----------
    timeframe : str
        Timeframe string (e.g., '1m', '5m', '15m', '1h', '1d')
    start_time : datetime
        Start datetime
    end_time : datetime
        End datetime
    
    Returns
    -------
    int
        Number of periods between start_time and end_time
    """
    minutes = get_timeframe_to_minutes(timeframe)
    total_minutes = (end_time - start_time).total_seconds() / 60
    total_periods = int(total_minutes // minutes)
    return total_periods

def get_timeframe_to_minutes( timeframe: str):
    """
    Convert timeframe string to minutes.
    
    Parameters
    ----------
    timeframe : str
        Timeframe string (e.g., '1m', '5m', '15m', '1h', '1d')
    
    Returns
    -------
    int
        Number of minutes for the timeframe. Defaults to 15 if timeframe not recognized.
    """
    periods_map = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360,
        '12h': 720, '1d': 1440,
    }
    return periods_map.get(timeframe, 15)


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


