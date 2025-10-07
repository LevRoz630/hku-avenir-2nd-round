## Backtester.run_permutation_backtest

### Summary
Runs observed backtest plus `permutations` shuffled price histories to form a null distribution for Sharpe and a p-value.

### Signature
`run_permutation_backtest(strategy, position_manager, start_date, end_date, time_step, market_type, permutations=100) -> Dict`

### Returns
- `p_value`: permutation test p-value on Sharpe.
- `observed_results`: result payload from unshuffled run (same schema as `run_backtest`).
- `sharpes`: list of Sharpe ratios over permutations.

### Data flow and storage
- Uses the same preloading as `run_backtest` to populate `HistoricalDataCollector` stores.
- Snapshots original dataframes per symbol; for i>0 runs, reshuffles rows (preserving timestamps for alignment semantics) and resets OMS/portfolio state:
  - `OMSClient.positions = {}`, `trade_history = []`, `balance['USDT'] = starting_balance`.
  - Clears `portfolio_values`, `returns`.
- Repeats the same loop and records returns per permutation.

### Notes
- Sharpe computed on per-step returns; p-value computed as the proportion of permuted Sharpes >= observed.


