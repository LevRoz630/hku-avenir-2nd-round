from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from oms_simulation import OMSClient
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from hist_data import HistoricalDataCollector
from format_utils import format_positions_table, format_balances_table
import plotly.graph_objects as go
import json
import copy
import inspect
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

logger = logging.getLogger(__name__)


def _run_single_permutation_worker(args):
    (permutation_idx, data_snapshots, base_symbols, market_type, start_date, end_date,
     time_step, aligned_start, starting_balance, historical_data_dir,
     strategy_class, strategy_init_args, position_manager_class, position_manager_init_args) = args
    
    try:
        data_manager = HistoricalDataCollector(historical_data_dir)
        if market_type == "spot":
            for sym in base_symbols:
                if sym in data_snapshots.get('spot', {}):
                    df = copy.deepcopy(data_snapshots['spot'][sym])
                    if permutation_idx > 0:
                        df = df.sample(frac=1, random_state=permutation_idx).reset_index(drop=True)
                        df["timestamp"] = data_snapshots['spot'][sym]["timestamp"]
                    data_manager.spot_ohlcv_data[sym] = df
        elif market_type == "futures":
            for sym in base_symbols:
                if sym in data_snapshots.get('index_future', {}):
                    df = copy.deepcopy(data_snapshots['index_future'][sym])
                    if permutation_idx > 0:
                        df = df.sample(frac=1, random_state=permutation_idx).reset_index(drop=True)
                        df["timestamp"] = data_snapshots['index_future'][sym]["timestamp"]
                    data_manager.perpetual_index_ohlcv_data[sym] = df
                if sym in data_snapshots.get('mark_future', {}):
                    df = copy.deepcopy(data_snapshots['mark_future'][sym])
                    if permutation_idx > 0:
                        df = df.sample(frac=1, random_state=permutation_idx + 1337).reset_index(drop=True)
                        df["timestamp"] = data_snapshots['mark_future'][sym]["timestamp"]
                    data_manager.perpetual_mark_ohlcv_data[sym] = df
        oms_client = OMSClient(historical_data_dir=historical_data_dir)
        oms_client.positions = {}
        oms_client.trade_history = []
        oms_client.balance = {"USDT": starting_balance}
        oms_client.set_current_time(aligned_start)
        oms_client.set_data_manager(data_manager)
        
        strategy = strategy_class(**strategy_init_args)
        strategy.start_time = aligned_start
        strategy.end_time = end_date
        
        position_manager = position_manager_class(**position_manager_init_args)
        
        portfolio_values = []
        iteration = 0
        
        while oms_client.current_time <= end_date:
            try:
                total_value = oms_client.update_portfolio_value()
                portfolio_values.append(total_value)
                
                orders = strategy.run_strategy(oms_client=oms_client, data_manager=data_manager)
                filtered = position_manager.filter_orders(orders=orders, oms_client=oms_client, data_manager=data_manager)
                
                if filtered is not None:
                    for order in filtered:
                        try:
                            oms_client.set_target_position(
                                order['symbol'], 
                                order['instrument_type'], 
                                order.get('value', 0.0), 
                                order['side']
                            )
                        except Exception as e:
                            logger.error(f"Error setting target position: {e}")
                            continue
                
                oms_client.set_current_time(oms_client.current_time + time_step)
                iteration += 1
                
            except Exception as e:
                logger.error(f"Error in backtest iteration {iteration}: {e}")
                oms_client.set_current_time(oms_client.current_time + time_step)
                continue
        returns = []
        for i in range(1, len(portfolio_values)):
            prev = portfolio_values[i-1]
            curr = portfolio_values[i]
            if prev > 0:
                returns.append((curr - prev) / prev)
        
        result = {
            'permutation_idx': permutation_idx,
            'returns': np.array(returns, dtype=float)
        }
        if permutation_idx == 0:
            drawdowns = []
            max_drawdown = 0
            peak = portfolio_values[0] if portfolio_values else 0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0
                drawdowns.append(drawdown)
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            sharpe_ratio = 0
            sortino_ratio = 0
            if returns:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    sharpe_ratio = mean_return / std_return
                downside_returns = [r for r in returns if r < 0]
                if downside_returns:
                    downside_std = np.std(downside_returns)
                    if downside_std > 0:
                        sortino_ratio = mean_return / downside_std
            
            result['observed_results'] = {
                'total_return': (portfolio_values[-1] / portfolio_values[0] - 1) if portfolio_values and portfolio_values[0] > 0 else 0,
                'returns': list(returns),
                'drawdowns': drawdowns,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'trade_history': [dict(t) for t in oms_client.trade_history],
                'final_balance': dict(oms_client.balance),
                'final_positions': oms_client.get_position()
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in permutation {permutation_idx}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'permutation_idx': permutation_idx,
            'returns': np.array([], dtype=float),
            'error': str(e)
        }


class Backtester:
    def __init__(self, historical_data_dir: str = "../hku-data/test_data"):
        """High-level backtest runner coordinating data, strategy and OMS.

        Responsibilities:
        - Ensure historical data availability and push into the collector cache
        - Iterate the clock, ask strategy for orders, run Position Manager filters
        - Send sized orders to OMS and track portfolio metrics

        Automatically sets up OMSClient and HistoricalDataCollector.
        Strategy and Position Manager are passed as an argument to run_backtest (strategy, position_manager)
        """
        self.data_manager = HistoricalDataCollector(historical_data_dir)
        self.oms_client = OMSClient(historical_data_dir=historical_data_dir)
        self.historical_data_dir = historical_data_dir
        self.portfolio_values = []
        self.returns = []
        self.drawdowns   = []
        self.max_drawdown = 0
        self.sharpe_ratio = 0
        self.sortino_ratio = 0
        self.trade_history = []
        self.final_balance = 0
        self.final_positions = []
        self.position_manager = None
        # Positions over time: list of dicts {timestamp: pd.Timestamp, exposures: {symbol: signed_notional}}
        self.position_exposures_history = []

        # For permutation tests: list of per-run return arrays; index 0 is the observed run
        self.permutation_returns = []
        
    def run_backtest(self, 
                        strategy: Any,
                        position_manager: Any,
                        start_date: datetime = None,
                        end_date: datetime = None,
                        time_step: timedelta = None,
                        market_type: str = None) -> Dict[str, Any]:
        """
        Execute a strategy over historical data and return performance.

        - Ensures data exists (downloads if missing)
        - Loads data into memory via HistoricalDataCollector
        - Aligns start_time to first available candle
        - Iterates time, lets strategy place orders via OMSClient
        - Aggregates portfolio values and computes metrics

        Args:
            strategy: Instantiated strategy object (must expose `symbols` and `lookback_days`)
            position_manager: Instantiated position manager object
            start_date: Start datetime (aligned to earliest available data if earlier)
            end_date: End datetime
            time_step: Time delta between backtest iterations (maps to data timeframe)
            market_type: "spot" or "futures"; futures loop uses index OHLCV and mark for execution
        Returns:
            Results dict with PnL series and summary metrics
        """
        try:
            self.position_manager = position_manager



            symbols = strategy.symbols
            # Ensure historical data exists for requested symbols; download if missing
            data_path = Path(self.historical_data_dir)
            base_symbols = [s.replace('-PERP', '') for s in symbols]
            data_path.mkdir(parents=True, exist_ok=True)

            if time_step is None:
                raise ValueError("Time step is required")

            desired_timeframe = self._time_step_to_timeframe(time_step)


            data_start_date = start_date - timedelta(days=strategy.lookback_days + 2)
        except Exception as e:
            logger.error(f"Error in run_backtest: {e}")
            return None

        for sym in base_symbols:
            try:
                if market_type == "spot":
                    self.data_manager.load_data_period(sym, desired_timeframe, 'ohlcv_spot', data_start_date, end_date, save_to_class=True, load_from_class=False, export=True)
                elif market_type == "futures":
                    # this is data for the backtest loop 
                    self.data_manager.load_data_period(sym, desired_timeframe, 'index_ohlcv_futures', data_start_date, end_date, save_to_class=True, load_from_class=False, export=True)

                    # this is data for price taking estiamtions when the position is opened  and risk management 
                    self.data_manager.load_data_period(sym, "15m", 'mark_ohlcv_futures', data_start_date, end_date, save_to_class=True, load_from_class=False, export=True)
                else:
                        raise ValueError(f"Invalid market type: {market_type}")
            except Exception as e:
                logger.error(f"Error loading data: {e} for {sym} for market type {market_type}")
                continue

        # Align start time to earliest available data so prices exist at t0
        earliest_ts = None
        for sym in base_symbols:
            for store in [self.data_manager.spot_ohlcv_data, self.data_manager.perpetual_mark_ohlcv_data, self.data_manager.perpetual_index_ohlcv_data]:
                df = store.get(sym)
                if df is not None and not df.empty:
                    first = df['timestamp'].min()  # first available candle for this store
                    if first.tz is not None:
                        first = first.tz_convert('UTC')
                    else:
                        first = first.tz_localize('UTC')
                    earliest_ts = first if earliest_ts is None or first < earliest_ts else earliest_ts

        aligned_start = start_date
        if earliest_ts is not None:
            # Ensure start_date is timezone-aware in UTC for comparison
            s = pd.Timestamp(start_date)
            e = pd.Timestamp(earliest_ts)
            if s.tz is None:
                s = s.tz_localize('UTC')
            else:
                s = s.tz_convert('UTC')
            if e.tz is None:
                e = e.tz_localize('UTC')
            else:
                e = e.tz_convert('UTC')
            if s < e:
                aligned_start = earliest_ts

        # Define start and end times for the strategy
        strategy.start_time = aligned_start
        strategy.end_time = end_date

        # Set the time for data fetching and orders that will be updated as strategy progresses
        self.oms_client.set_current_time(strategy.start_time)
        self.oms_client.set_data_manager(self.data_manager)

        # Reset per-run histories
        self.position_exposures_history = []

        # Run backtest
        iteration = 0

        while self.oms_client.current_time <= end_date:
            try:
                # Revalue portfolio at the current timestamp
                total_value = self.oms_client.update_portfolio_value()
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

                orders = strategy.run_strategy(oms_client=self.oms_client, data_manager=self.data_manager)

                # we pass the oms and data manager to the position manager so it would have the most up to date information
                filtered_orders = self.position_manager.filter_orders(orders=orders, oms_client=self.oms_client, data_manager=self.data_manager)

                if filtered_orders is not None:
                    for order in filtered_orders:
                        try:
                            self.oms_client.set_target_position(
                                order['symbol'],
                                order['instrument_type'],
                                order.get('value', 0.0),
                                order['side']
                            )
                        except Exception as e:
                            logger.error(f"Error setting target position: {e}")
                            continue

                # Snapshot signed notional exposures for all symbols at this time
                try:
                    exposures = {}
                    for symbol, pos in self.oms_client.positions.items():
                        instrument_type = pos.get('instrument_type', 'future' if symbol.endswith('-PERP') else 'spot')
                        current_price = self.oms_client.get_current_price(symbol, instrument_type)
                        if not current_price:
                            continue
                        qty = float(pos.get('quantity', 0.0))
                        # signed notional: qty * price (SHORTs will have negative qty)
                        exposures[symbol] = qty * current_price
                    self.position_exposures_history.append({
                        'timestamp': pd.Timestamp(self.oms_client.current_time),
                        'exposures': exposures
                    })
                except Exception as e:
                    logger.debug(f"Error snapshotting exposures: {e}")

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
            'drawdowns': self.drawdowns,
            'sharpe_ratio': self.sharpe_ratio,
            'trade_history': self.oms_client.trade_history,
            'final_balance': self.oms_client.balance,
            'final_positions': self.oms_client.get_position()
        }

        return results



    def run_permutation_backtest(self, 
                                 strategy: Any,
                                 position_manager: Any,
                                 start_date: datetime = None,
                                 end_date: datetime = None,
                                 time_step: timedelta = None,
                                 market_type: str = None,
                                 permutations: int = 100,
                                 max_workers: int = None):
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)

        data_path = Path(self.historical_data_dir)
        symbols = strategy.symbols
        base_symbols = [s.replace('-PERP', '') for s in symbols]
        data_path.mkdir(parents=True, exist_ok=True)
        if time_step is None:
            raise ValueError("Time step is required")
        desired_timeframe = self._time_step_to_timeframe(time_step)

        data_start_date = start_date - timedelta(days=strategy.lookback_days + 2)
        for sym in base_symbols:
            if market_type == "spot":
                self.data_manager.load_data_period(sym, desired_timeframe, 'ohlcv_spot', data_start_date, end_date, save_to_class=True, load_from_class=False, export=True)
            elif market_type == "futures":
                self.data_manager.load_data_period(sym, desired_timeframe, 'index_ohlcv_futures', data_start_date, end_date, save_to_class=True, load_from_class=False, export=True)
                self.data_manager.load_data_period(sym, "15m", 'mark_ohlcv_futures', data_start_date, end_date, save_to_class=True, load_from_class=False, export=True)
        starting_balance = float(self.oms_client.balance['USDT'])

        data_snapshots = {
            'spot': {},
            'index_future': {},
            'mark_future': {}
        }

        for sym in base_symbols:
            if market_type == "spot":
                df = self.data_manager.spot_ohlcv_data.get(sym)
                if df is not None:
                    data_snapshots['spot'][sym] = df.copy()
            elif market_type == "futures":
                df_idx = self.data_manager.perpetual_index_ohlcv_data.get(sym)
                if df_idx is not None:
                    data_snapshots['index_future'][sym] = df_idx.copy()
                df_mark = self.data_manager.perpetual_mark_ohlcv_data.get(sym)
                if df_mark is not None:
                    data_snapshots['mark_future'][sym] = df_mark.copy()
            else:
                raise ValueError(f"Invalid market type: {market_type}")

        earliest_ts = None
        for sym in base_symbols:
            for store in [self.data_manager.spot_ohlcv_data, self.data_manager.perpetual_mark_ohlcv_data, self.data_manager.perpetual_index_ohlcv_data]:
                df = store.get(sym)
                if df is not None and not df.empty:
                    first = df['timestamp'].min()
                    if first.tz is None:
                        first = first.tz_localize('UTC')
                    else:
                        first = first.tz_convert('UTC')
                    earliest_ts = first if earliest_ts is None or first < earliest_ts else earliest_ts
        aligned_start = start_date
        if earliest_ts is not None and start_date < earliest_ts:
            aligned_start = earliest_ts

        strategy_class = strategy.__class__
        strategy_init_args = self._extract_init_args(strategy)
        position_manager_class = position_manager.__class__
        position_manager_init_args = self._extract_init_args(position_manager)

        worker_args_list = []
        for i in range(permutations + 1):
            args = (
                i,  # permutation_idx
                data_snapshots,
                base_symbols,
                market_type,
                start_date,
                end_date,
                time_step,
                aligned_start,
                starting_balance,
                self.historical_data_dir,
                strategy_class,
                strategy_init_args,
                position_manager_class,
                position_manager_init_args
            )
            worker_args_list.append(args)

        logger.info(f"Running {permutations + 1} permutations ({permutations} shuffled + 1 observed) with {max_workers} workers")
        permutation_results = {}
        observed_results = None

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_run_single_permutation_worker, args): args[0] for args in worker_args_list}
            
            completed = 0
            for future in as_completed(future_to_idx):
                completed += 1
                try:
                    result = future.result()
                    idx = result['permutation_idx']
                    permutation_results[idx] = result
                    
                    if idx == 0:
                        observed_results = result.get('observed_results')
                    
                    logger.info(f"Completed permutation {idx} ({completed}/{permutations + 1})")
                except Exception as e:
                    idx = future_to_idx[future]
                    logger.error(f"Permutation {idx} failed: {e}")
                    permutation_results[idx] = {
                        'permutation_idx': idx,
                        'returns': np.array([], dtype=float)
                    }

        # Aggregate results in order
        permutation_returns = []
        for i in range(permutations + 1):
            if i in permutation_results:
                permutation_returns.append(permutation_results[i]['returns'])
            else:
                permutation_returns.append(np.array([], dtype=float))

        if len(permutation_returns) > 0 and len(permutation_returns[0]) > 0:
            def _sharpe(arr: np.ndarray) -> float:
                arr = np.asarray(arr, dtype=float)
                if arr.size == 0:
                    return np.nan
                mu = np.nanmean(arr)
                sigma = np.nanstd(arr, ddof=1)
                return mu / sigma if sigma > 0 and np.isfinite(mu) else np.nan

            def _sortino(arr: np.ndarray) -> float:
                arr = np.asarray(arr, dtype=float)
                if arr.size == 0:
                    return np.nan
                mu = np.nanmean(arr)
                downside_returns = arr[arr < 0]
                if len(downside_returns) == 0:
                    return np.nan
                downside_std = np.nanstd(downside_returns, ddof=1)
                return mu / downside_std if downside_std > 0 and np.isfinite(mu) else np.nan

            T_obs = _sharpe(permutation_returns[0])
            sharpe_list = []
            sortino_list = []
            for r in permutation_returns:
                s = _sharpe(r)
                if np.isfinite(s):
                    sharpe_list.append(s)
                s = _sortino(r)
                if np.isfinite(s):
                    sortino_list.append(s)
            sharpes = np.array(sharpe_list, dtype=float)
            B = sharpes.size
            p_value = (1 + np.sum(sharpes >= T_obs)) / (B + 1) if B > 0 and np.isfinite(T_obs) else np.nan
            return {
                'p_value': float(p_value) if np.isfinite(p_value) else np.nan,
                'observed_results': observed_results,
                'sharpes': sharpes.tolist(),
                'sortinos': sortino_list,
            }

        return {
            'p_value': np.nan,
            'observed_results': observed_results,
            'sharpes': [],
            'sortinos': [],
        }

    def _extract_init_args(self, instance: Any) -> Dict[str, Any]:
        init_signature = inspect.signature(instance.__class__.__init__)
        args_dict = {}
        
        for param_name, param in init_signature.parameters.items():
            if param_name == 'self':
                continue
            
            if param.default == inspect.Parameter.empty and not hasattr(instance, param_name):
                continue
            
            if hasattr(instance, param_name):
                try:
                    value = getattr(instance, param_name)
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        args_dict[param_name] = value
                    elif isinstance(value, Path):
                        args_dict[param_name] = str(value)
                except Exception:
                    continue
        
        return args_dict

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
        
        # Calculate max drawdown and store drawdown series
        peak = self.portfolio_values[0]
        self.max_drawdown = 0
        self.drawdowns = []
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            self.drawdowns.append(drawdown)
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        
        # Calculate Sharpe ratio (use period returns; assume 24 periods/day for 1h)
        if self.returns:
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns)
            if std_return > 0:
                self.sharpe_ratio = mean_return / std_return
    
    def plot_portfolio_value(self):
        """Plot backtest results"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=self.portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title="Portfolio Value",
            xaxis_title="Time",
            yaxis_title="Portfolio Value",
            hovermode='x unified',
            showlegend=True
        )
        fig.show()

    def plot_drawdown(self):
        """Plot drawdown"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=self.drawdowns,
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2)
        ))
        fig.update_layout(
            title="Drawdown",
            xaxis_title="Time",
            yaxis_title="Drawdown",
            hovermode='x unified',
            showlegend=True
        )
        fig.show()

    def plot_returns(self):
        """Plot returns"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=self.returns,
            mode='lines',
            name='Returns',
            line=dict(color='green', width=2)
        ))
        fig.update_layout(
            title="Returns",
            xaxis_title="Time",
            yaxis_title="Returns",
            hovermode='x unified',
            showlegend=True
        )
        fig.show()
    
    def plot_positions(self):
        """Plot signed notional exposures by symbol over time as a heatmap."""
        if not self.position_exposures_history:
            return
        # Build DataFrame: rows=time, cols=symbols, values=signed notional
        times = [entry['timestamp'] for entry in self.position_exposures_history]
        all_symbols = set()
        for entry in self.position_exposures_history:
            all_symbols.update(entry['exposures'].keys())
        all_symbols = sorted(list(all_symbols))
        data = []
        for entry in self.position_exposures_history:
            row = [entry['exposures'].get(sym, 0.0) for sym in all_symbols]
            data.append(row)
        import numpy as np
        import plotly.colors as pc
        z = np.array(data, dtype=float)
        fig = go.Figure(data=go.Heatmap(
            z=z.T,
            x=times,
            y=all_symbols,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(
            title="Signed Notional Exposures Over Time",
            xaxis_title="Time",
            yaxis_title="Symbol",
        )
        fig.show()

    def save_results(self, results: Dict[str, Any], filename: str):
        """Save backtest results"""
        # Convert non-serializable fields (e.g., datetimes) in trade_history
        serializable = dict(results)
        try:
            serializable['trade_history'] = [
                {**t, 'timestamp': (t.get('timestamp').isoformat() if isinstance(t.get('timestamp'), datetime) else t.get('timestamp'))}
                for t in results.get('trade_history', [])
            ]
        except Exception:
            serializable['trade_history'] = results.get('trade_history', [])
        with open(filename+".json", "w") as f:
            json.dump(serializable, f)

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