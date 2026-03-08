from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
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
import sys

logger = logging.getLogger(__name__)


def _run_single_permutation_worker(args):
    (permutation_idx, data_snapshots, base_symbols, market_type, start_date, end_date,
     time_step, aligned_start, starting_balance, historical_data_dir,
     strategy_class, strategy_init_args, position_manager_class, position_manager_init_args) = args
    
    try:
        data_manager = HistoricalDataCollector(historical_data_dir)
        if market_type != "futures":
            raise ValueError("Only futures market type is supported")

        for sym in base_symbols:
            if sym in data_snapshots.get('mark_future', {}):
                df = copy.deepcopy(data_snapshots['mark_future'][sym])
                if permutation_idx > 0:
                    df = df.sample(frac=1, random_state=permutation_idx + 1337).reset_index(drop=True)
                    df["timestamp"] = data_snapshots['mark_future'][sym]["timestamp"]
                data_manager.perpetual_mark_ohlcv_data[sym] = df
            if sym in data_snapshots.get('index_future', {}):
                df = copy.deepcopy(data_snapshots['index_future'][sym])
                if permutation_idx > 0:
                    df = df.sample(frac=1, random_state=permutation_idx).reset_index(drop=True)
                    df["timestamp"] = data_snapshots['index_future'][sym]["timestamp"]
                data_manager.perpetual_index_ohlcv_data[sym] = df

        oms_client = OMSClient(historical_data_dir=historical_data_dir)
        oms_client.positions = {}
        oms_client.trade_history = []
        oms_client.balance = {"USDT": starting_balance}
        oms_client.set_current_time(aligned_start)
        oms_client.set_data_manager(data_manager)
        
        strategy = strategy_class(**strategy_init_args)
        strategy.start_time = aligned_start
        strategy.end_time = end_date

        self.last_backtest_start = aligned_start
        self.last_backtest_end = end_date
        
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

        # Regime data for plotting
        self.regime_data = None
        self.last_backtest_start: Optional[datetime] = None
        self.last_backtest_end: Optional[datetime] = None
        self.requested_backtest_start: Optional[datetime] = None
        self.requested_backtest_end: Optional[datetime] = None

        # Price cache for performance optimization
        self.price_cache = {}

    def get_cached_price(self, symbol: str, timestamp: datetime, instrument_type: str = None) -> float:
        instrument_type = instrument_type or 'future'
        cache_key = f"{symbol}_{instrument_type}_{timestamp.isoformat()}"

        if cache_key in self.price_cache:
            return self.price_cache[cache_key]

        # Price not cached, fetch it
        if not self.oms_client:
            return None

        # Temporarily set OMS time to get price
        original_time = self.oms_client.current_time
        self.oms_client.current_time = timestamp
        price = self.oms_client.get_current_price(symbol, instrument_type)
        self.oms_client.current_time = original_time

        # Cache the price
        if price is not None:
            self.price_cache[cache_key] = price

        return price

    def _update_portfolio_value_cached(self) -> float:
        if not self.oms_client:
            return 0.0

        total_value = self.oms_client.balance['USDT']
        current_time = self.oms_client.current_time

        for symbol, pos in self.oms_client.positions.items():
            current_price = self.get_cached_price(symbol, current_time, 'future')
            if not current_price:
                continue

            quantity = float(pos.get('quantity', 0.0))
            side = pos.get('side', 'LONG')
            entry = float(pos.get('entry_price', 0.0))


            if abs(quantity) > 0:
                if side == 'LONG':
                    unrealized = quantity * (current_price - entry)
                elif side == 'SHORT':
                    unrealized = quantity * (entry - current_price)
                else:
                    unrealized = 0.0
                total_value += unrealized

            # Maintain notional value field for reporting only
            position_value = abs(quantity) * current_price
            pos['value'] = position_value
            

        return total_value

    def run_backtest(self,
                        strategy: Any,
                        position_manager: Any,
                        start_date: datetime = None,
                        end_date: datetime = None,
                        time_step: timedelta = None,
                        market_type: str = None) -> Dict[str, Any]:
        self.requested_backtest_start = start_date
        self.requested_backtest_end = end_date
        try:
            self.position_manager = position_manager



            symbols = strategy.symbols
            data_path = Path(self.historical_data_dir)
            base_symbols = [s.replace('-PERP', '').replace('-USDT', '') for s in symbols]
            data_path.mkdir(parents=True, exist_ok=True)

            if time_step is None:
                raise ValueError("Time step is required")

            desired_timeframe = self._time_step_to_timeframe(time_step)


            data_start_date = start_date - timedelta(days=strategy.lookback_days + 2)
        except Exception as e:
            logger.error(f"Error in run_backtest: {e}")
            return None

        # Preload data
        extended_end_date = end_date + timedelta(days=1)
        required_data = []

        if market_type != "futures":
            raise ValueError("Only futures market type is supported")

        for sym in base_symbols:
            required_data.append((sym, "15m", 'mark_ohlcv_futures', data_start_date, extended_end_date))
            required_data.append((sym, "15m", 'index_ohlcv_futures', data_start_date, extended_end_date))
            if desired_timeframe != "15m":
                required_data.append((sym, desired_timeframe, 'mark_ohlcv_futures', data_start_date, extended_end_date))

        seen = set()
        unique_required = []
        for item in required_data:
            key = (item[0], item[1], item[2])
            if key in seen:
                continue
            seen.add(key)
            unique_required.append(item)

        requested_start = start_date

        for data_spec in unique_required:
            sym, timeframe, data_type, preload_start, preload_end = data_spec
            try:
                result = self.data_manager.load_data_period(sym, timeframe, data_type, preload_start, preload_end, save_to_class=True, load_from_class=False, export=True)
                if result is None or result.empty:
                    logger.warning(f"No {data_type} data available for {sym} ({timeframe}) in period {preload_start} to {preload_end}")
            except Exception as e:
                logger.error(f"Error preloading {data_type} data for {sym}: {e}")
                continue

        # Align start time to earliest available data
        earliest_ts = None
        for sym in base_symbols:
            stores_to_check = [
                self.data_manager.perpetual_mark_ohlcv_data,
                self.data_manager.perpetual_index_ohlcv_data
            ]

            for store in stores_to_check:
                df = store.get(sym)
                if df is not None and not df.empty:
                    first = df['timestamp'].min()  # first available candle for this store
                    if first.tz is not None:
                        first = first.tz_convert('UTC')
                    else:
                        first = first.tz_localize('UTC')
                    earliest_ts = first if earliest_ts is None or first < earliest_ts else earliest_ts

        aligned_start = requested_start
        if earliest_ts is not None:
            s = pd.Timestamp(requested_start)
            e = pd.Timestamp(earliest_ts)
            if s.tz is None:
                s = s.tz_localize('UTC')
            else:
                s = s.tz_convert('UTC')
            if e.tz is None:
                e = e.tz_localize('UTC')
            else:
                e = e.tz_convert('UTC')
            aligned_start = max(s, e)

        if aligned_start.tzinfo is None:
            aligned_start = aligned_start.tz_localize('UTC')

        strategy.start_time = aligned_start
        strategy.end_time = end_date

        self.oms_client.set_current_time(strategy.start_time)
        self.oms_client.set_data_manager(self.data_manager)

        self.position_exposures_history = []
        self.price_cache.clear()
        iteration = 0
        total_iterations = int((end_date - aligned_start) / time_step) + 1
        logger.info(f"Starting backtest with ~{total_iterations} iterations")

        while self.oms_client.current_time <= end_date:
            try:
                total_value = self._update_portfolio_value_cached()
                self.portfolio_values.append(total_value)

                orders = strategy.run_strategy(oms_client=self.oms_client, data_manager=self.data_manager)
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

                # Log
                current_time = self.oms_client.current_time.strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"{current_time} | Portfolio: ${total_value:.2f}")

                # Format orders
                if filtered_orders and len(filtered_orders) > 0:
                    try:
                        orders_data = []
                        for order in filtered_orders:
                            orders_data.append({
                                'Symbol': order['symbol'],
                                'Type': order['instrument_type'],
                                'Side': order['side'],
                                'Value': f"${order.get('value', 0.0):.2f}"
                            })

                        if orders_data:
                            orders_df = pd.DataFrame(orders_data)
                            logger.info(f"Orders executed:\n{orders_df.to_string(index=False)}")
                    except Exception as e:
                        logger.debug(f"Could not format orders table: {e}")

                # Snapshot exposures every 10 iterations
                if iteration % 10 == 0:
                    try:
                        exposures = {}
                        current_time = self.oms_client.current_time
                        for symbol, pos in self.oms_client.positions.items():
                            current_price = self.get_cached_price(symbol, current_time, 'future')
                            if not current_price:
                                continue
                            qty = float(pos.get('quantity', 0.0))
                            side = pos.get('side', 'LONG')
                            # signed notional: positive for LONG, negative for SHORT
                            signed_notional = qty * current_price if side == 'LONG' else -qty * current_price
                            exposures[symbol] = signed_notional
                        self.position_exposures_history.append({
                            'timestamp': pd.Timestamp(current_time),
                            'exposures': exposures
                        })
                    except Exception as e:
                        logger.debug(f"Error snapshotting exposures: {e}")

                self.oms_client.set_current_time(self.oms_client.current_time + time_step)
                iteration += 1
                    
            except Exception as e:
                logger.error(f"Error in backtest iteration {iteration}: {e}")
                self.oms_client.set_current_time(self.oms_client.current_time + time_step)
                continue

        logger.info("Backtest complete")

        self.calculate_performance_metrics()
        final_portfolio_value = self.portfolio_values[-1] if self.portfolio_values else self.oms_client.balance['USDT']

        results = {
            'total_return': (self.portfolio_values[-1] / self.portfolio_values[0] - 1) if self.portfolio_values else 0,
            'returns': self.returns,
            'max_drawdown': self.max_drawdown,
            'drawdowns': self.drawdowns,
            'sharpe_ratio': self.sharpe_ratio,
            'trade_history': self.oms_client.trade_history,
            'final_balance': self.oms_client.balance,
            'final_portfolio_value': final_portfolio_value,
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
        base_symbols = [s.replace('-PERP', '').replace('-USDT', '') for s in symbols]
        data_path.mkdir(parents=True, exist_ok=True)
        if time_step is None:
            raise ValueError("Time step is required")
        desired_timeframe = self._time_step_to_timeframe(time_step)

        if market_type != "futures":
            raise ValueError("Only futures market type is supported")

        data_start_date = start_date - timedelta(days=strategy.lookback_days + 2)
        extended_end_date = end_date + timedelta(days=1)

        for sym in base_symbols:
            self.data_manager.load_data_period(sym, "15m", 'mark_ohlcv_futures', data_start_date, extended_end_date, save_to_class=True, load_from_class=False, export=True)
            self.data_manager.load_data_period(sym, "15m", 'index_ohlcv_futures', data_start_date, extended_end_date, save_to_class=True, load_from_class=False, export=True)
            if desired_timeframe != "15m":
                self.data_manager.load_data_period(sym, desired_timeframe, 'mark_ohlcv_futures', data_start_date, extended_end_date, save_to_class=True, load_from_class=False, export=True)

        starting_balance = float(self.oms_client.balance['USDT'])

        data_snapshots = {
            'mark_future': {},
            'index_future': {}
        }

        for sym in base_symbols:
            df_mark = self.data_manager.perpetual_mark_ohlcv_data.get(sym)
            if df_mark is not None:
                data_snapshots['mark_future'][sym] = df_mark.copy()
            df_idx = self.data_manager.perpetual_index_ohlcv_data.get(sym)
            if df_idx is not None:
                data_snapshots['index_future'][sym] = df_idx.copy()

        earliest_ts = None
        for sym in base_symbols:
            for store in [self.data_manager.perpetual_mark_ohlcv_data, self.data_manager.perpetual_index_ohlcv_data]:
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
        if len(self.portfolio_values) < 2:
            return
        self.returns = []
        for i in range(1, len(self.portfolio_values)):
            prev = self.portfolio_values[i-1]
            curr = self.portfolio_values[i]
            if prev > 0:
                self.returns.append((curr - prev) / prev)
        
        # Max drawdown
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
        
        # Sharpe ratio
        if self.returns:
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns)
            if std_return > 0:
                self.sharpe_ratio = mean_return / std_return
    
    def plot_portfolio_value(self):
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

    def load_regime_data(self, regime_labels_path: str = None):
        if regime_labels_path is None:
            # Default path based on hypothesis testing artifacts
            repo_root = Path(__file__).parent.parent.parent
            regime_labels_path = repo_root / "hypothesis_testing" / "cointegration" / "artifacts" / "hmm_labels_15m_mark.parquet"

        try:
            return self._generate_regime_data(regime_labels_path)
        except Exception as e:
            logger.error(f"Error generating regime data: {e}")
            return False

    def _generate_regime_data(self, output_path: Path) -> bool:
        try:
            # Add hypothesis testing path for imports
            sys.path.append(str(Path(__file__).parent.parent.parent / "hypothesis_testing" / "cointegration"))

            # Import required functions
            from hmm_regimes import train_and_persist_labels

            # Get mark price data from the backtester's data manager
            mark_data = self.data_manager.perpetual_mark_ohlcv_data

            if not mark_data:
                return False

            # Combine all loaded symbols
            price_frames = []
            for symbol, df in mark_data.items():
                if df is not None and not df.empty:
                    try:
                        # Ensure timestamp is datetime index
                        if 'timestamp' in df.columns:
                            df = df.set_index('timestamp')
                        if not isinstance(df.index, pd.DatetimeIndex):
                            df.index = pd.to_datetime(df.index)

                        close_col = f"{symbol}_close"
                        price_series = df[['close']].rename(columns={'close': close_col})
                        price_frames.append(price_series)
                    except Exception as e:
                        logger.warning(f"Error processing {symbol} data: {e}")
                        continue

            if not price_frames:
                logger.error("No valid price data found in backtester")
                return False

            # Combine all symbols
            price_data = pd.concat(price_frames, axis=1, join='outer').dropna(how='all')

            # Generate regime labels
            bars_per_day = 96  # 15m bars * 24 hours
            meta = {
                'generated_at': datetime.now().isoformat(),
                'source': 'backtester_loaded_data',
                'timeframe': '15m',
                'price_type': 'mark',
                'bars_per_day': bars_per_day,
                'symbols': len([col for col in price_data.columns if col.endswith('_close')])
            }

            labels_df = train_and_persist_labels(
                cointegration_data=price_data,
                bars_per_day=bars_per_day,
                output_path=output_path,
                meta=meta
            )

            # Load the generated data
            self.regime_data = labels_df
            if labels_df.empty:
                logger.info("No regime data generated - insufficient data quality or time period too short for HMM analysis")
            return True

        except Exception as e:
            logger.error(f"Error generating regime data: {e}")
            return False

    def plot_regimes(self):
        if self.regime_data is None or self.regime_data.empty:
            return  # Silently skip if no regime data available

        regime_df = self.regime_data

        # Use requested backtest dates for regime filtering
        filter_start = self.requested_backtest_start or self.last_backtest_start
        filter_end = self.requested_backtest_end or self.last_backtest_end

        if isinstance(regime_df.index, pd.DatetimeIndex) and filter_start and filter_end:
            # Convert backtest dates to pandas Timestamps in UTC
            start_ts = pd.Timestamp(filter_start)
            end_ts = pd.Timestamp(filter_end)

            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize('UTC')
            elif start_ts.tzinfo != timezone.utc:
                start_ts = start_ts.tz_convert('UTC')

            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize('UTC')
            elif end_ts.tzinfo != timezone.utc:
                end_ts = end_ts.tz_convert('UTC')

            # Match timezone awareness for comparison
            if regime_df.index.tz is None:
                start_cmp = start_ts.tz_localize(None)
                end_cmp = end_ts.tz_localize(None)
            else:
                start_cmp = start_ts.tz_convert(regime_df.index.tz)
                end_cmp = end_ts.tz_convert(regime_df.index.tz)

            # Filter the data
            filtered = regime_df.loc[(regime_df.index >= start_cmp) & (regime_df.index <= end_cmp)]

            if not filtered.empty:
                regime_df = filtered

        # Get regime columns (ending with _hmm_state)
        regime_cols = [col for col in regime_df.columns if col.endswith('_hmm_state')]
        if not regime_cols or regime_df.empty:
            return  # Silently skip if no valid regime data

        # Extract symbol names from column names
        symbols = [col.replace('_hmm_state', '') for col in regime_cols]

        # Create heatmap
        colorscale = [[0, 'green'], [1, 'red']]
        fig = go.Figure(data=go.Heatmap(
            z=regime_df[regime_cols].values.T,
            x=regime_df.index,
            y=symbols,
            colorscale=colorscale,
            zmin=0,
            zmax=1
        ))

        fig.update_layout(
            title="HMM Regimes (Green=Low, Red=High)",
            xaxis_title="Time",
            yaxis_title="Symbol"
        )
        fig.show()

    def save_results(self, results: Dict[str, Any], filename: str):
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
        print(f"\nTotal Return: {results['total_return']:.2%}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Final Portfolio Value: ${results['final_portfolio_value']:.2f}")
        print(f"Cash Balance: {results['final_balance']}")
        print(f"Trades: {len(results['trade_history'])}")


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