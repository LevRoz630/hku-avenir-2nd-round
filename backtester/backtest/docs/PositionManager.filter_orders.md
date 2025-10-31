## PositionManager.filter_orders

### Summary
Validates and sizes raw strategy orders. Risk-screens by short-horizon vol, sizes by inverse-vol weights under a 10% USDT budget, and enforces cash constraints.

### Signature
`filter_orders(orders, oms_client, data_manager) -> List[Dict] | None`

### Steps and data usage
1. Split `CLOSE` orders to always allow exits.
2. `_close_risky_orders` loads `mark_ohlcv_futures` (15m, 4h window) from `HistoricalDataCollector.perpetual_mark_ohlcv_data[base_symbol]`; sets `value=0` if scaled vol > 0.1.
3. `_set_weights` loads 1d of 15m mark data and computes inverse-vol weights; allocates 10% of `balance['USDT']` proportionally.
4. If sized opens exceed cash, return only `CLOSE` orders; else combined result.


