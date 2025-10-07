## OMSClient.get_total_portfolio_value

### Summary
Computes portfolio value: cash + spot notionals + futures unrealized PnL.

### Signature
`get_total_portfolio_value() -> float`

### Data flow and storage
- Iterates `positions` and fetches current prices via `get_current_price` (backed by `HistoricalDataCollector`).
- Futures contribute unrealized PnL only and maintain `value` for reporting.
- Spot contributes full notional; updates per-position `value`.


