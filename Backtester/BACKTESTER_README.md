# Backtester for Trading Strategies

A comprehensive backtesting framework that allows you to test your existing OMS-based trading strategies using historical data with minimal code changes.

## Features

- **OMS-Compatible Interface**: Works with existing strategies without modification
- **Historical Data Support**: Uses collected historical data from `hist_data.py`
- **Multiple Strategy Support**: Compatible with demo.py and fundingrate2.py strategies
- **Performance Metrics**: Calculates returns, drawdown, Sharpe ratio, and more
- **Trade Analysis**: Tracks and analyzes all executed trades
- **Visualization**: Generates performance charts and comparisons

## Quick Start

### 1. Collect Historical Data

First, collect historical data using the existing data collector:

```python
from hist_data import HistoricalDataCollector

# Collect 30 days of data
collector = HistoricalDataCollector(days=30, type="future")
symbols = ["BTC-USDT-PERP", "ETH-USDT-PERP", "SOL-USDT-PERP"]

# Collect different types of data
collector.collect_historical_ohlcv(symbols, timeframe='1h')
collector.collect_historical_funding_rates(symbols)
collector.collect_historical_open_interest(symbols, timeframe='1h')
collector.collect_historical_trades(symbols)
```

### 2. Run Backtest with Existing Strategy

```python
from backtester import Backtester
from strategy_adapter import run_backtest_with_strategy
from datetime import datetime, timedelta

# Initialize backtester
backtester = Backtester(
    historical_data_dir="historical_data",
    initial_balance=10000.0,
    symbols=["BTC-USDT-PERP", "ETH-USDT-PERP"]
)

# Run backtest with demo strategy
results = run_backtest_with_strategy(
    backtester,
    strategy_name="demo",  # or "funding_rate"
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now(),
    time_step=timedelta(hours=1)
)

# Print results
backtester.print_results(results)
```

### 3. Run Complete Analysis

```bash
python backtest_example.py
```

This will:
- Collect historical data (if not available)
- Run both demo and funding rate strategies
- Compare performance
- Generate charts
- Save results to JSON

## Supported Strategies

### Demo Strategy (`demo.py`)
- **Logic**: Long top 2 performers, short bottom 2 performers based on 7-day returns
- **Usage**: `strategy_name="demo"`

### Funding Rate Strategy (`fundingrate2.py`)
- **Logic**: Uses funding rate signals for position sizing
- **Usage**: `strategy_name="funding_rate"`

## OMS-Compatible Functions

The backtester provides these functions that work exactly like the real OMS:

```python
# Account functions
balance = strategy.get_account_balance()
positions = strategy.get_current_positions()

# Data functions
historical_data = strategy.get_historical_data()
funding_data = strategy.get_funding_rate_data(periods=10)
open_interest_data = strategy.get_open_interest_data()

# Trading functions
strategy.push_target_positions(positions_dict, type="future")
strategy.show_account_detail()
```

## Performance Metrics

The backtester calculates:

- **Total Return**: Overall portfolio return
- **Max Drawdown**: Maximum peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return metric
- **Trade Count**: Number of executed trades
- **Daily Returns**: Distribution of daily returns

## File Structure

```
data_59q897/
├── backtester.py              # Main backtester class
├── strategy_adapter.py        # Strategy compatibility layer
├── backtest_example.py        # Complete example script
├── BACKTESTER_README.md       # This file
├── historical_data/           # Collected historical data
│   ├── future_BTC-USDT-PERP_1h_30d.csv
│   ├── future_BTC-USDT-PERP_funding_rates_30d.csv
│   └── ...
└── backtest_results.json      # Saved results
```

## Custom Strategies

To use your own strategy:

1. **Create Strategy Class**: Follow the same pattern as demo.py
2. **Use OMS Functions**: Use the provided OMS-compatible functions
3. **Adapt for Backtester**: Use the strategy adapter

```python
from strategy_adapter import StrategyAdapter

# Your custom strategy
class MyStrategy:
    def __init__(self):
        self.oms_client = None
        self.current_time = None
    
    def run_strategy(self):
        # Your strategy logic using OMS functions
        balance = self.get_account_balance()
        positions = self.get_current_positions()
        # ... strategy logic ...
        self.push_target_positions(target_positions)

# Adapt for backtester
adapter = StrategyAdapter(MyStrategy, backtester)
strategy = adapter.create_adapted_strategy()
```

## Advanced Usage

### Custom Time Steps

```python
# Run with different time steps
results = backtester.run_backtest(
    strategy_class=MyStrategy,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    time_step=timedelta(minutes=15)  # 15-minute steps
)
```

### Multiple Symbol Testing

```python
# Test with different symbol sets
backtester = Backtester(
    symbols=["BTC-USDT-PERP", "ETH-USDT-PERP", "SOL-USDT-PERP", "BNB-USDT-PERP"],
    initial_balance=50000.0
)
```

### Performance Analysis

```python
# Access detailed results
results = backtester.run_backtest(strategy_class=MyStrategy)

# Portfolio values over time
portfolio_values = results['portfolio_values']

# Trade history
trades = results['trade_history']

# Performance metrics
total_return = results['total_return']
max_drawdown = results['max_drawdown']
sharpe_ratio = results['sharpe_ratio']
```

## Troubleshooting

### No Historical Data
- Run `python backtest_example.py` to collect data automatically
- Or manually collect using `hist_data.py`

### Strategy Import Errors
- Ensure strategy files are in the correct directory
- Check Python path and imports

### Performance Issues
- Reduce time step for faster execution
- Use fewer symbols or shorter time periods
- Optimize strategy logic

## Example Output

```
==================================================
BACKTEST RESULTS
==================================================
Total Return: 12.34%
Max Drawdown: -5.67%
Sharpe Ratio: 1.23
Final Balance: {'USDT': 11234.0}
Number of Trades: 45
==================================================
```

## Next Steps

1. **Collect Data**: Run data collection for your desired symbols and timeframes
2. **Test Strategies**: Run backtests with different strategies
3. **Analyze Results**: Compare performance and optimize
4. **Deploy**: Use successful strategies in live trading

For more examples, see `backtest_example.py` which demonstrates the complete workflow.
