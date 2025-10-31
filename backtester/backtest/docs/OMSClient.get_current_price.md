## OMSClient.get_current_price

### Summary
Returns current price from cached data up to `current_time + timestep_delta` to emulate execution after the decision timestamp.

### Signature
`get_current_price(symbol, instrument_type=None) -> float | None`

### Data flow and storage
- Spot: reads `HistoricalDataCollector.spot_ohlcv_data[base_symbol]`.
- Futures: reads `HistoricalDataCollector.perpetual_mark_ohlcv_data[base_symbol]`.
- Applies timestamp filtering `<= current_time + min_candle_diff` and returns the latest `close`.

### Notes
- Returns `None` when no data is available; callers should handle gracefully.


