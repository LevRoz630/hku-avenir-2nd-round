# Usage Guide

## Configuration

The system now reads OMS credentials from `oms.txt` (which is gitignored for security):

```
# oms.txt format:
# Line 1: (empty)
# Line 2: test
# Line 3: test_token_here
# Line 4: (empty)
# Line 5: prod
# Line 6: prod_token_here
```

## Running the Strategy

### Basic Usage
```bash
# Run in test mode with real-time monitoring
python demo_english.py

# Run in production mode
python demo_english.py --mode prod

# Run strategy once and exit (no real-time monitoring)
python demo_english.py --run-once
```

### Data Collection
```bash
# Default collection (15-minute data for 7 days)
python collect_data_flexible.py

# Hourly data for longer periods
python collect_data_flexible.py --timeframe 1h --days 30

# 15-minute data for medium periods
python collect_data_flexible.py --timeframe 15m --days 7

# 1-minute data for short periods
python collect_data_flexible.py --timeframe 1m --days 1

# Custom data length
python collect_data_flexible.py --timeframe 15m --days 14   # 2 weeks
python collect_data_flexible.py --timeframe 1h --days 90    # 3 months
python collect_data_flexible.py --timeframe 1m --days 3     # 3 days of 1-minute data
```

## Command Line Options

- `--mode {test,prod}`: Choose OMS environment (default: test)
- `--run-once`: Run strategy once and exit (useful for testing)

## Files Structure

- `config.py` - Configuration management (reads oms.txt)
- `demo_english.py` - Main trading strategy
- `data_collector.py` - Data collection for testing
- `collect_test_data.py` - Script to collect test data
- `oms.txt` - OMS credentials (gitignored)
- `test_data/` - Collected data files (gitignored)

## Strategy Behavior

1. **Initialization**: Loads OMS credentials and connects to Binance
2. **Data Collection**: Fetches 7 days of historical data
3. **Strategy Execution**: Runs daily at 08:00
   - Calculates 7-day returns for all symbols
   - Longs top 2 performers, shorts bottom 2
   - Allocates 90% of USDT balance equally
4. **Real-time Monitoring**: Subscribes to 1-minute BTC/USDT K-lines
5. **Account Updates**: Refreshes balance/positions hourly

## Safety Features

- All sensitive data (tokens, collected data) is gitignored
- Test/prod mode separation
- Rate limiting to prevent API abuse
- Error handling and logging
- Graceful shutdown with Ctrl+C
