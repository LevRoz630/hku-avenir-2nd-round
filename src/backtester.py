from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from oms_simulation import BacktesterOMS
import logging
import numpy as np
from pathlib import Path
from hist_data import HistoricalDataCollector
from format_utils import format_positions_table, format_balances_table
from position_manager import PositionManager

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, historical_data_dir: str = "../hku-data/test_data"):
        self.data_manager = HistoricalDataCollector(historical_data_dir)
        self.oms_client = BacktesterOMS(historical_data_dir=historical_data_dir)
        self.historical_data_dir = historical_data_dir
        self.portfolio_values = []
        self.returns = []
        self.max_drawdown = 0
        self.sharpe_ratio = 0
        self.trade_history = []
        self.final_balance = 0
        self.final_positions = []
        self.position_manager = PositionManager()

        # For permutation tests: list of per-run return arrays; index 0 is the observed run
        self.permutation_returns = []
        
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
        

        # Ensure historical data exists for requested symbols; download if missing
        data_path = Path(self.historical_data_dir)
        base_symbols = [s.replace('-PERP', '') for s in symbols]
        data_path.mkdir(parents=True, exist_ok=True)

        if time_step is None:
            raise ValueError("Time step is required")

        desired_timeframe = self._time_step_to_timeframe(time_step)


        data_start_date = start_date - timedelta(days=strategy.lookback_days + 2)
        dm = self.data_manager
        

        for sym in base_symbols:
            if market_type == "spot":
                dm.load_data_period(sym, desired_timeframe, 'ohlcv_spot', data_start_date, end_date, export=True)
            elif market_type == "futures":
                # this is data for the backtest loop 
                dm.load_data_period(sym, desired_timeframe, 'index_ohlcv_futures', data_start_date, end_date, export=True)

                # this is data for price taking estiamtions when the position is opened  and risk management 
                dm.load_data_period(sym, "15m", 'mark_ohlcv_futures', data_start_date, end_date, export=True)
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
        self.oms_client.set_current_time(strategy.start_time)
        self.oms_client.set_timestep(time_step)

        # Run backtest
        iteration = 0

        while self.oms_client.current_time <= end_date:
            try:
                # Revalue portfolio at the current timestamp
                total_value = self.oms_client.get_total_portfolio_value()
                self.portfolio_values.append(total_value)
                logger.info(f"Total Portfolio Value: {total_value}")
                # Pretty-print balances and positions
                try:
                    positions_tbl = format_positions_table(self.oms_client.get_position())
                    logger.info("\nPositions:\n" + positions_tbl)

                except Exception as _e:
                    # Fallback to raw summary on any formatting error
                    summary = self.oms_client.get_position_summary()
                    logger.info(f"Position Summary: {summary}")

                orders = strategy.run_strategy(current_time=self.oms_client.current_time, data_manager=self.data_manager)
                
                filtered_orders = self.position_manager.filter_orders(orders=orders, oms_client=self.oms_client)

                
                for order in filtered_orders:
                    try:
                        self.oms_client.set_target_position(order['symbol'], order['value'], order['side'], order['instrument_type'])
                    except Exception as e:
                        logger.error(f"Error setting target position: {e}")
                        continue
                # Move to next time step
                self.oms_client.set_current_time(self.oms_client.current_time + time_step)
                iteration += 1
                    
            except Exception as e:
                logger.error(f"Error in backtest iteration {iteration}: {e}")
                self.oms_client.set_current_time(self.oms_client.current_time + time_step)
                continue
        
        # Calculate final performance metrics
        self.calculate_performance_metrics()
        # Return results
        results = {
            'total_return': (self.portfolio_values[-1] / self.portfolio_values[0] - 1) if self.portfolio_values else 0,
            'returns': self.returns,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'trade_history': self.oms_client.trade_history,
            'final_balance': self.oms_client.balance,
            'final_positions': self.oms_client.get_position()
        }

        # Cleanly close live clients
        try:
            self.oms_client.data_manager.close()
        except Exception:
            pass
        return results



    def run_permutation_backtest(self, strategy: Any, symbols: List[str], start_date: datetime = None, end_date: datetime = None, time_step: timedelta = None, market_type: str = None, permutations: int = 100):
        """
        Run a backtest for a given strategy and symbols with different permutations of the symbols
        """
        
        # Ensure historical data exists for requested symbols; download if missing
        data_path = Path(self.historical_data_dir)
        base_symbols = [s.replace('-PERP', '') for s in symbols]
        data_path.mkdir(parents=True, exist_ok=True)
        desired_timeframe = self._time_step_to_timeframe(time_step)


        data_start_date = start_date - timedelta(days=strategy.lookback_days + 2)
        dm = self.oms_client.data_manager
        
        for i in range(permutations+1):
            print(f"permutation {i} of {permutations}")
            if i == 0:
                for sym in symbols:
                    if market_type == "spot":
                        dm.load_data_period(sym, desired_timeframe, 'ohlcv_spot', data_start_date, end_date, export=True)
                    elif market_type == "futures":
                        # this is data for the backtest loop 
                        dm.load_data_period(sym, desired_timeframe, 'index_ohlcv_futures', data_start_date, end_date, export=True)

                        # this is data for price taking estiamtions when the position is opened  and risk management 
                        dm.load_data_period(sym, "15m", 'mark_ohlcv_futures', data_start_date, end_date, export=True)
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

            if i > 0:
                for symbol in symbols:
                    df  = dm.spot_ohlcv_data.get(symbol) 
                    if df is not None and not df.empty:
                        dm.spot_ohlcv_data[symbol] = df.sample(frac=1).reset_index(drop=True)

                    df  = dm.perpetual_index_ohlcv_data.get(symbol)
                    if df is not None and not df.empty:
                        dm.perpetual_index_ohlcv_data[symbol] = df.sample(frac=1).reset_index(drop=True)

            # Define start and end times for the strategy
            strategy.start_time = aligned_start
            strategy.end_time = end_date

            # Set the time for data fetching and orders that will be updated as strategy progresses
            self.oms_client.set_current_time(strategy.start_time)
            
            if time_step is None:
                time_step = timedelta(minutes=15)

            self.oms_client.set_timestep(time_step)

            # Run backtest
            iteration = 0

            while self.oms_client.current_time <= end_date:
                try:
                    # Revalue portfolio at the current timestamp
                    total_value = self.oms_client.get_total_portfolio_value()
                    self.portfolio_values.append(total_value)
                    logger.info(f"Total Portfolio Value: {total_value}")
                    # Pretty-print balances and positions
                    try:
                        positions_tbl = format_positions_table(self.oms_client.get_position())
                        logger.info("\nPositions:\n" + positions_tbl)

                    except Exception as _e:
                        # Fallback to raw summary on any formatting error
                        summary = self.oms_client.get_position_summary()
                        logger.info(f"Position Summary: {summary}")

                    strategy.run_strategy(current_time=self.oms_client.current_time, data_manager=self.data_manager)

                    # Move to next time step
                    self.oms_client.set_current_time(self.oms_client.current_time + time_step)
                    iteration += 1
                        
                except Exception as e:
                    logger.error(f"Error in backtest iteration {iteration}: {e}")
                    self.oms_client.set_current_time(self.oms_client.current_time + time_step)
                    continue
            
            print(f"permutation {i} of {permutations} complete")
            # Calculate final performance metrics and record this run's returns
            self.calculate_performance_metrics()
            self.permutation_returns.append(np.array(self.returns, dtype=float))
            self.portfolio_values = []
            self.returns = []
            self.trade_history = []
            self.final_positions = []

            for sym in symbols:
                if market_type == "spot":
                    dm.load_data_period(sym, desired_timeframe, 'ohlcv_spot', data_start_date, end_date, export=True)
                elif market_type == "futures":
                    # this is data for the backtest loop 
                    dm.load_data_period(sym, desired_timeframe, 'index_ohlcv_futures', data_start_date, end_date, export=True)

                    # this is data for price taking estiamtions when the position is opened  and risk management 
                    dm.load_data_period(sym, "15m", 'mark_ohlcv_futures', data_start_date, end_date, export=True)
                else:
                    raise ValueError(f"Invalid market type: {market_type}")


        print(f"permutation returns: {len(self.permutation_returns)}")
        # After all permutations, compute permutation p-value on Sharpe ratios

        if len(self.permutation_returns) > 0:
            def _sharpe(arr: np.ndarray) -> float:
                arr = np.asarray(arr, dtype=float)
                if arr.size == 0:
                    return np.nan
                mu = np.nanmean(arr)
                sigma = np.nanstd(arr, ddof=1)
                return mu / sigma if sigma > 0 and np.isfinite(mu) else np.nan

            T_obs = _sharpe(self.permutation_returns[0])
            sharpe_list = []
            for r in self.permutation_returns:
                s = _sharpe(r)
                if np.isfinite(s):
                    sharpe_list.append(s)
            sharpes = np.array(sharpe_list, dtype=float)
            B = sharpes.size
            p_value = (1 + np.sum(sharpes >= T_obs)) / (B + 1) if B > 0 and np.isfinite(T_obs) else np.nan
            return p_value

        return None

    def calculate_performance_metrics(self):
        """Calculate performance metrics"""
        if len(self.portfolio_values) < 2:
            return
            
        # Calculate period returns (based on time_step granularity)
        self.returns = []
        for i in range(1, len(self.portfolio_values)):
            prev = self.portfolio_values[i-1]
            curr = self.portfolio_values[i]
            if prev > 0:
                self.returns.append((curr - prev) / prev)
        
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
        if self.returns:
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns)
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


    # Derive timeframe string from time_step for consistent collection/loading
    def _time_step_to_timeframe(self, ts: timedelta) -> str:
        if ts is None:
            raise ValueError("Time step is None")
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