## Backtester.run_backtest

### Summary
Execute a strategy over historical data and return performance metrics. Orchestrates the loop, data access, order filtering, OMS execution, and metric computation.

### Signature
`run_backtest(strategy, position_manager, start_date, end_date, time_step, market_type) -> Dict`

### Parameters
- `strategy`: exposes `symbols: List[str]`, `lookback_days: int`, and `run_strategy(oms_client, data_manager)`.
- `position_manager`: `PositionManager` with `filter_orders(orders, oms_client, data_manager)`.
- `start_date`, `end_date`: UTC datetimes; start aligns to earliest available data if needed.
- `time_step`: `timedelta`; mapped to data timeframe via `_time_step_to_timeframe`.
- `market_type`: `"spot" | "futures"`.

### Returns
`{ total_return, returns, max_drawdown, sharpe_ratio, trade_history, final_balance, final_positions }`

### Data flow and storage
- Preload data by symbol:
  - Spot: `ohlcv_spot` into `HistoricalDataCollector.spot_ohlcv_data[symbol]`.
  - Futures: `index_ohlcv_futures` (loop) into `perpetual_index_ohlcv_data[symbol]` and `mark_ohlcv_futures` (execution/risk) into `perpetual_mark_ohlcv_data[symbol]`.
- Align start time to earliest available candle across loaded stores.
- Set OMS context: `current_time`, `timestep`, `data_manager`.
- Loop per step:
  1) Revalue via `OMSClient.get_total_portfolio_value()`; track `portfolio_values`.
  2) Call `strategy.run_strategy(oms_client, data_manager)` to emit raw orders.
  3) Pass to `PositionManager.filter_orders(...)` for risk and sizing.
  4) Execute each order with `OMSClient.set_target_position(...)`, which updates `positions`, `balance` (spot), and `trade_history`.
  5) Advance `current_time += time_step`.

### Notes
- Futures portfolio value adds unrealized PnL only; spot adds full notional value.
- `get_current_price` sources mark prices for futures and spot OHLCV for spot, up to `current_time + timestep_delta`.

### Minimal example
See `backtest/example.py` for a runnable setup with `HoldStrategy` and `PositionManager`.


