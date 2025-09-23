from datetime import datetime, timedelta
from typing import List, Dict, Any
from oms_simulation import BacktesterOMS
import logging
import numpy as np
from pathlib import Path
from hist_data import HistoricalDataCollector
from format_utils import format_positions_table, format_balances_table

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, historical_data_dir: str = "../hku-data/test_data"):
        self.historical_data_dir = historical_data_dir
        self.portfolio_values = []
        self.daily_returns = []
        self.max_drawdown = 0
        self.sharpe_ratio = 0
        self.trade_history = []
        self.final_balance = 0
        self.final_positions = []

    def run_backtest(self, 
                        strategy: Any,
                        symbols: List[str],
                        start_date: datetime = None,
                        end_date: datetime = None,
                        time_step: timedelta = None,
                        market_type: str = None) -> Dict[str, Any]:
        """
        Execute a strategy over historical data and return performance.

        - Ensures data exists (downloads if missing)
        - Loads data into memory via HistoricalDataCollector
        - Aligns start_time to first available candle
        - Iterates time, lets strategy place orders via OMS
        - Aggregates portfolio values and computes metrics

        Args:
            strategy: Instantiated strategy object
            symbols: Symbols to backtest (supports -PERP for perps)
            start_date: Start datetime (aligned to first data if earlier)
            end_date: End datetime
            time_step: Time delta between backtest iterations
            market_type: Market type to backtest (spot or futures)
        Returns:
            Results dict with PnL series and summary metrics
        """
        # Derive timeframe string from time_step for consistent collection/loading
        def _time_step_to_timeframe(ts: timedelta) -> str:
            if ts is None:
                return '1h'
            minutes = int(ts.total_seconds() // 60)
            if minutes == 1:
                return '1m'
            if minutes == 5:
                return '5m'
            if minutes == 15:
                return '15m'
            if minutes == 30:
                return '30m'
            if minutes == 60:
                return '1h'
            return '15m'

        # Ensure historical data exists for requested symbols; download if missing
        data_path = Path(self.historical_data_dir)
        base_symbols = [s.replace('-PERP', '') for s in symbols]
        data_path.mkdir(parents=True, exist_ok=True)
        desired_timeframe = _time_step_to_timeframe(time_step)

        dm = strategy.oms_client.data_manager
        for sym in base_symbols:
            if market_type == "spot":
                print(f"DEBUG load_data_period sym={sym} data_type=ohlcv_spot, start_date={start_date}, end_date={end_date}")
                dm.load_data_period(sym, desired_timeframe, 'ohlcv_spot', start_date, end_date, export=True)
            elif market_type == "futures":
                print(f"DEBUG load_data_period sym={sym} data_type=mark_ohlcv_futures / data_type=index_ohlcv_futures, start_date={start_date}, end_date={end_date}")
                dm.load_data_period(sym, desired_timeframe, 'mark_ohlcv_futures', start_date, end_date, export=True)
                dm.load_data_period(sym, desired_timeframe, 'index_ohlcv_futures', start_date, end_date, export=True)
            else:
                raise ValueError(f"Invalid market type: {market_type}")

        # Align start time to earliest available data so prices exist at t0
        earliest_ts = None
        for sym in base_symbols:
            for store in [dm.spot_ohlcv_data, dm.perpetual_mark_ohlcv_data, dm.perpetual_index_ohlcv_data]:
                df = store.get(sym)
                if df is not None and not df.empty:
                    first = df['timestamp'].min()  # first available candle for this store
                    earliest_ts = first if earliest_ts is None or first < earliest_ts else earliest_ts
        aligned_start = start_date
        if earliest_ts is not None and start_date < earliest_ts:
            aligned_start = earliest_ts

        # Define start and end times for the strategy
        strategy.start_time = aligned_start
        strategy.end_time = end_date

        # Set the time for data fetching and orders that will be updated as strategy progresses
        strategy.oms_client.set_current_time(strategy.start_time)
        if time_step is None:
            time_step = timedelta(minutes=15)
        # Run backtest
        iteration = 0

        while strategy.oms_client.current_time <= end_date:
            try:
                # Revalue portfolio at the current timestamp
                total_value = strategy.oms_client.get_total_portfolio_value()
                self.portfolio_values.append(total_value)
                logger.info(f"Total Portfolio Value: {total_value}")
                # Pretty-print balances and positions
                try:
                    balances_tbl = format_balances_table(strategy.oms_client.get_account_balance())
                    positions_tbl = format_positions_table(strategy.oms_client.get_position())
                    logger.info("\nBalances:\n" + balances_tbl)
                    logger.info("\nPositions:\n" + positions_tbl)
                except Exception as _e:
                    # Fallback to raw summary on any formatting error
                    summary = strategy.oms_client.get_position_summary()
                    logger.info(f"Position Summary: {summary}")

                strategy.run_strategy()
                
                
                # Move to next time step
                strategy.oms_client.set_current_time(strategy.oms_client.current_time + time_step)
                iteration += 1
                
                # Log progress every 24 iterations (daily if hourly steps)
                if iteration % 24 == 0:
                    logger.info(f"Backtest progress: {strategy.oms_client.current_time} (Iteration {iteration})")
                    
            except Exception as e:
                logger.error(f"Error in backtest iteration {iteration}: {e}")
                strategy.oms_client.set_current_time(strategy.oms_client.current_time + time_step)
                continue
        
        # Calculate final performance metrics
        self.calculate_performance_metrics()
        
        # Return results
        return {
            'portfolio_values': self.portfolio_values,
            'daily_returns': self.daily_returns,
            'total_return': (self.portfolio_values[-1] / self.portfolio_values[0] - 1) if self.portfolio_values else 0,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'trade_history': strategy.oms_client.trade_history,
            'final_balance': strategy.oms_client.get_account_balance(),
            'final_positions': strategy.oms_client.get_position()
        }

    def calculate_performance_metrics(self):
        """Calculate performance metrics"""
        if len(self.portfolio_values) < 2:
            return
            
        # Calculate period returns (based on time_step granularity)
        self.period_returns = []
        for i in range(1, len(self.portfolio_values)):
            prev = self.portfolio_values[i-1]
            curr = self.portfolio_values[i]
            if prev > 0:
                self.period_returns.append((curr - prev) / prev)
        
        # Calculate max drawdown
        peak = self.portfolio_values[0]
        self.max_drawdown = 0
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        
        # Calculate Sharpe ratio (use period returns; assume 24 periods/day for 1h)
        if self.period_returns:
            mean_return = np.mean(self.period_returns)
            std_return = np.std(self.period_returns)
            if std_return > 0:
                self.sharpe_ratio = mean_return / std_return * np.sqrt(252*24*4)

    def print_results(self, results: Dict[str, Any]):
        """Print backtest results in a formatted way"""
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Final Balance: {results['final_balance']}")
        print(f"Number of Trades: {len(results['trade_history'])}")
        print("="*50)

