## OMSClient.set_target_position

### Summary
Places/adjusts a position using a USDT target value; supports `CLOSE` to exit. Handles same-side adds and flips.

### Signature
`set_target_position(symbol, instrument_type, target_value, position_side) -> Dict`

### Parameters
- `symbol`: e.g., `BTC-USDT` (futures internally normalized to base for data lookup).
- `instrument_type`: `"future"`.
- `target_value`: USDT to deploy at current price (spot subtracts cash; futures do not).
- `position_side`: `"LONG" | "SHORT" | "CLOSE"`.

### Data flow and storage
- Reads price via `get_current_price` backed by `HistoricalDataCollector` stores.
- Updates in-memory `positions[symbol]` with quantity, side, entry_price, value, and cumulative PnL.
- Spot flows: adjusts `balance['USDT']` for principal on open/add/flip; futures adjust balance only by realized PnL on close/flip.
- Appends an immutable record to `trade_history` with timestamp, symbol, side, quantity, value, entry price, and post-trade balance snapshot.

### Errors
- Raises on missing price or insufficient `balance['USDT']`.


