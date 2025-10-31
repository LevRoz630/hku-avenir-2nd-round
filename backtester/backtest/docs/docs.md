## Backtest Engine: Structure and Workflow

### Components
- **Backtester (`src/backtester.py`)**: Orchestrates the time loop, calls `strategy.run_strategy(...)`, passes orders through `PositionManager.filter_orders(...)`, and executes via `OMSClient.set_target_position(...)`. Computes returns, drawdown, Sharpe, and aggregates results.
- **OMSClient (`src/oms_simulation.py`)**: Tracks `balance['USDT']`, `positions`, and `trade_history`. Provides `get_current_price`, `set_target_position`, `close_position`, `get_total_portfolio_value`, and reporting helpers.
- **HistoricalDataCollector (`src/hist_data.py`)**: Loads/caches OHLCV and related series via `load_data_period(symbol, timeframe, data_type, start, end)`.
- **Strategy (e.g., `backtest/strategies/v1_hold.py`)**: Implements `run_strategy(oms_client, data_manager) -> List[order]`.
- **PositionManager (`src/position_manager.py`)**: `filter_orders(orders, oms_client, data_manager)` risk-screens, sizes by inverse-vol, and enforces budget before OMS execution.

### Data flow (per timestep)
1) Backtester revalues portfolio via `OMSClient.get_total_portfolio_value()`; logs positions.
2) Strategy emits raw orders: `run_strategy(oms_client, data_manager) -> List[Dict]`.
3) PositionManager:
   - Risk screen on last 4h of 15m mark OHLCV; if scaled vol > 0.1, set `value=0`.
   - Size remaining orders by inverse-vol weights under a 10% USDT budget.
   - If sized opens exceed cash, return only any `CLOSE` orders; otherwise return combined list.
4) Backtester executes each order via `OMSClient.set_target_position(symbol, instrument_type, value, side)`.
5) Advance `current_time` by `time_step`; repeat until `end_date`.

### Order schema (after PositionManager)
```json
{
  "symbol": "BTC-USDT",                // or base + -PERP-normalized internally for futures
  "instrument_type": "future",
  "side": "LONG" | "SHORT" | "CLOSE",
  "value": 1234.56                      // USDT notional; PM supplies, it is possible to rewrite for the strategy to supply it as well
}
```
 ### Data storage map
 - `HistoricalDataCollector.spot_ohlcv_data[symbol]`: spot loop and pricing source when `market_type="spot"`.
 - `HistoricalDataCollector.perpetual_index_ohlcv_data[symbol]`: futures loop timing/prices.
 - `HistoricalDataCollector.perpetual_mark_ohlcv_data[symbol]`: futures execution pricing and PM risk inputs.
 - `HistoricalDataCollector.funding_rates_data[symbol]`: available for strategies/PM if needed.
 - `HistoricalDataCollector.open_interest_data[symbol]`: optional risk/signal input.
 - `OMSClient.positions[symbol]`: live in-memory state per symbol (quantity, side, entry_price, value, pnl, instrument_type).
 - `OMSClient.trade_history`: immutable list of executed actions with timestamp and post-trade balance snapshot.

### Data used by market_type
- **spot**: loop prices from `ohlcv_spot` (timeframe from `time_step`).
- **futures**: loop on `index_ohlcv_futures` (derived timeframe) and use `mark_ohlcv_futures` for execution pricing and risk metrics.

Timeframe derivation: `Backtester._time_step_to_timeframe(...)` maps `time_step` to `{'1m','5m','15m','30m','1h'}`; defaults to `15m` if unmatched.

### OMS semantics (key points)
- `set_target_position`: interprets `value` as USDT to deploy at current price; creates/adjusts/close/flip positions.
- Futures do not move principal cash on open/adjust; spot subtracts cash. Portfolio value for futures adds unrealized PnL only; spot adds full notional value.

### Example: Hold strategy + PositionManager
- Hold emits once per symbol: `{symbol, instrument_type='future', side='LONG'}`; PM sizes under 10% budget with inverse-vol; Backtester executes; OMS maintains state.

Minimal usage (see `backtest/example.py`):
```python
from datetime import datetime, timedelta, timezone
from backtester import Backtester
from position_manager import PositionManager
from strategies.v1_hold import HoldStrategy

bt = Backtester(historical_data_dir="./historical_data")
strategy = HoldStrategy(symbols=["BTC-USDT","ETH-USDT"], lookback_days=0)
pm = PositionManager()
start_date = datetime.now(timezone.utc)-timedelta(days=30)
end_date = datetime.now(timezone.utc)
results = bt.run_backtest(
    strategy=strategy,
    position_manager=pm,
    start_date=start_date,
    end_date=end_date,
    time_step=timedelta(hours=1),
    market_type="futures",
)


results = backtester.run_permutation_backtest(
    strategy=strategy,
    position_manager=pm,
    start_date=start_date,
    end_date=end_date,
    time_step=timedelta(days = 1),
    market_type="futures",
    permutations=3,
)
print("p_value:", results.get("p_value"))

```


