
from ccxt import binance as binance_async
import pandas as pd
from ccxt.pro import binance
from datetime import datetime
symbol = "ETHUSDT"
perp = "ETH-USDT-PERP"
limit = 1000
target_date = datetime(2025, 9, 11)  # September 11, 2025
timestamp = int(target_date.timestamp() * 1000)  
timeframe = "1m"
# Historical
ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=timestamp, limit=1000)
print(ohlcv)
# Live (CCXT Pro)
ohlcv = binance.watch_ohlcv(symbol, timeframe)
print("Live (CCXT Pro)")
print(ohlcv)