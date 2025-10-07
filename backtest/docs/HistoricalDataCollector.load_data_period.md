## HistoricalDataCollector.load_data_period

### Summary
Unified wrapper to load cached or collected data into in-memory stores for spot/futures series.
If not cached or saved, collects the data frpom api with batch limiting and retries

### Signature
`load_data_period(symbol, timeframe, data_type, start_date, end_date, export=False) -> DataFrame`

### Data storage
- Populates store by `data_type`:
  - `ohlcv_spot` -> `spot_ohlcv_data[symbol]`
  - `index_ohlcv_futures` -> `perpetual_index_ohlcv_data[symbol]`
  - `mark_ohlcv_futures` -> `perpetual_mark_ohlcv_data[symbol]`
  - `funding_rates` -> `funding_rates_data[symbol]`
  - `open_interest` -> `open_interest_data[symbol]`
  - `trades_futures` -> `perpetual_trades_data[symbol]`

### Notes
- Normalizes timestamps to tz-aware UTC and filters to `[start_date, end_date]`.
- Cache-first via `load_cached_window`; falls back to collection on cache miss.


