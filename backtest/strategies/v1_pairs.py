from typing import List, Any, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from oms_simulation import BacktesterOMS
# Module-level config setter to pass parameters from the runner script
_PAIRS_CONFIG: List[Dict] = []

def set_pairs_config(pairs_config: List[Dict]):
    global _PAIRS_CONFIG
    _PAIRS_CONFIG = pairs_config


class PairTradingStrategy:
    def __init__(self, symbols: List[str], historical_data_dir: str, lookback: int):
        self.symbols = symbols
        self.oms_client = BacktesterOMS(historical_data_dir=historical_data_dir)
        self.lookback = lookback
        # Build runtime state per pair
        self.pairs = []
        self.historical_data = pd.DataFrame()
        self.current_time = None
        self.start_time = None
        self.end_time = None

        for cfg in _PAIRS_CONFIG:
            base_a, base_b = cfg['legs']
            # Legs include -USDT already; append -PERP only for futures
            self.pairs.append({
                'a_symbol': f"{base_a}-PERP",
                'b_symbol': f"{base_b}-PERP",
                'lookback_days': cfg.get('lookback_days', 7),
                'entry_z': cfg.get('entry_z', 1.5),
                'exit_z': cfg.get('exit_z', 0.5),
                'max_alloc_frac': cfg.get('max_alloc_frac', 0.2),
                'is_open': False,
                'current_side': None,  # 'long_spread' or 'short_spread'
                'last_beta': 1.0,
            })

    def _get_daily_closes(self, base_symbol: str, lookback: int) -> pd.Series:
        """Return daily close series up to current time using mark OHLCV.

        Falls back to resampling 15m/1h data to daily closes if 1d is not present.
        """
        dm = self.oms_client.data_manager
        # Use existing collector method which returns the past n days, set export to False as we don't want to use saved data over and over again

        # This conditional should reduce the time we spend collecting data as we only collect full forty days at the start and then check if we need to collect more
        
        # Fetch a rolling window to ensure we have prior days before current day
        window_days = max(self.lookback + 2, 3)
        window_start = self.oms_client.current_time - pd.Timedelta(days=window_days)
        start_for_load = max(self.start_time, window_start) if self.start_time is not None else window_start
        end_for_load = self.oms_client.current_time
        print(f"DEBUG data_params base={base_symbol} tf=15m start={start_for_load} end={end_for_load}")
        df = dm.load_data_period(base_symbol, timeframe='15m', data_type='index_ohlcv_futures', start_date=start_for_load, end_date=end_for_load)
        if df is None or df.empty:
            print(f"DEBUG data_empty base={base_symbol} tf=15m start={self.start_time} end={self.oms_client.current_time}")
            return pd.Series(dtype=float)
        df = df.copy()
        df = df.sort_values('timestamp')
        print(f"DEBUG raw_loaded base={base_symbol} rows={len(df)} ts_range=({df['timestamp'].min()} -> {df['timestamp'].max()})")
        df = df.set_index('timestamp')
        # Resample to daily closes
        daily = df['close'].resample('1D').last().dropna()
        print(f"DEBUG resampled_daily base={base_symbol} days={len(daily)} idx_range=({daily.index.min()} -> {daily.index.max()})")

        daily = daily.iloc[-self.lookback:]
        print(f"DEBUG sliced_lookback base={base_symbol} days={len(daily)} idx_range=({daily.index.min()} -> {daily.index.max()})")
        # Align to prior day close relative to current time
        cutoff = self.oms_client.current_time.normalize()  # midnight of current day
        daily = daily[daily.index < cutoff]
        if not daily.empty:
            print(f"DEBUG filtered_pre_now base={base_symbol} days={len(daily)} last_idx={daily.index.max()} cutoff={cutoff}")
        return daily

    def _compute_beta_and_z(self, a_base: str, b_base: str):
        """Compute dynamic beta on daily log-prices and z-score like offline.

        - Beta from regression of log(y) on log(x) over last `lookback` daily closes (excluding current day)
        - Current spread and historical spreads computed on raw prices using that beta
        - z = (current_spread - mean(historical_spreads)) / std(historical_spreads)
        """
        a_series = self._get_daily_closes(a_base, self.lookback)
        b_series = self._get_daily_closes(b_base, self.lookback)
        if a_series.empty or b_series.empty:
            print(f"DEBUG series_empty a_len={len(a_series)} b_len={len(b_series)}")
            return None, None, None
        # Align on common dates
        df = pd.concat([a_series.rename('a'), b_series.rename('b')], axis=1).dropna()
        print(f"DEBUG aligned_len={len(df)} a_range=({a_series.index.min()} -> {a_series.index.max()}) b_range=({b_series.index.min()} -> {b_series.index.max()})")
        if len(df) < self.lookback + 1:
            print(f"DEBUG insufficient_aligned len={len(df)} needed={self.lookback + 1}")
            return None, None, None
        # Use last (lookback + 1) daily points; last one is "current" bar
        window = df.iloc[-(self.lookback + 1):]
        hist = window.iloc[:-1]
        curr = window.iloc[-1]
        # Regression on log prices over history window
        y = np.log(hist['a'].values)
        x = np.log(hist['b'].values).reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        beta = float(reg.coef_[0])
        # Historical spreads using raw prices with this beta
        hist_spreads = hist['a'].values - beta * hist['b'].values
        spread_mean = float(hist_spreads.mean())
        spread_std = float(hist_spreads.std())
        if spread_std <= 0:
            return beta, 0.0, 0.0
        current_spread = float(curr['a'] - beta * curr['b'])
        z = (current_spread - spread_mean) / spread_std
        print(f"DEBUG beta_z a={a_base} b={b_base} beta={beta:.4f} z={float(z):.3f} std={spread_std:.6f}")
        return beta, float(z), spread_std


    def run_strategy(self):
        # Portfolio value for sizing
        total_equity = float(self.oms_client.get_total_portfolio_value() or 0.0)
        print(f"DEBUG run start_time={self.start_time} now={self.oms_client.current_time} total_equity={total_equity}")

        # Enforce 90% allocation cap across all open pairs
        max_alloc_total = 0.90 * total_equity
        # Estimate current deployed margin notionally as sum of per-leg USDT allocations kept in state
        # We will track per-pair intended alloc to cap new entries
        if not hasattr(self, '_alloc_state'):
            self._alloc_state = {id(p): 0.0 for p in self.pairs}
        # Track last entry day per pair to enforce one entry per day
        if not hasattr(self, '_last_entry_day'):
            self._last_entry_day = {id(p): None for p in self.pairs}

        # Only make entry/exit decisions once per day using daily signals
        now = self.oms_client.current_time
        is_decision_time = True

        for p in self.pairs:
            a_base = p['a_symbol'].replace('-PERP', '')
            b_base = p['b_symbol'].replace('-PERP', '')

            # Compute daily beta/z only at decision time; otherwise hold
            if not is_decision_time:
                continue

            print(f"DEBUG compute_params pair=({a_base},{b_base}) lookback={self.lookback}")
            beta, z, std = self._compute_beta_and_z(a_base, b_base)
            if beta is None:
                print(f"DEBUG skip_pair reason=no_beta_z pair=({a_base},{b_base})")
                continue
            p['last_beta'] = beta

            a_sym = p['a_symbol']
            b_sym = p['b_symbol']

            instrument_type = 'future'

            # Determine desired action
            enter = abs(z) > p['entry_z']
            exit_ = abs(z) < p['exit_z'] and p['is_open']
            stop = abs(z) > (3.0 * p['entry_z']) and p['is_open']

            print(f"DEBUG decision t={now} pair=({a_base},{b_base}) z={z} enter={abs(z) > p['entry_z']} exit={abs(z) < p['exit_z'] and p['is_open']} stop={abs(z) > (3.0 * p['entry_z']) and p['is_open']} open={p['is_open']}")
            if enter:
                # Enforce one entry per calendar day
                day_key = (now.year, now.month, now.day)
                if self._last_entry_day.get(id(p)) == day_key:
                    # Already entered today; skip
                    pass
                else:
                    # Dollar allocation per pair (total across both legs)
                    alloc_cap = max(0.0, min(1.0, p['max_alloc_frac'])) * total_equity
                    # Scale with z-score up to 2x at z >= 2*entry
                    scale = min(abs(z) / max(p['entry_z'], 1e-6), 2.0)
                    total_pair_usdt = alloc_cap * scale
                    # Beta-weighted split: A gets 1, B gets |beta|
                    w_a = 1.0
                    w_b = abs(beta)
                    denom = max(w_a + w_b, 1e-8)
                    a_usdt = total_pair_usdt * (w_a / denom)
                    b_usdt = total_pair_usdt * (w_b / denom)

                    # Check portfolio cap before opening
                    current_deployed = sum(self._alloc_state.values())
                    print(f"DEBUG alloc_check deployed={current_deployed} add={total_pair_usdt} cap={max_alloc_total}")
                    if current_deployed + total_pair_usdt <= max_alloc_total:
                        if z > 0:
                            if p['current_side'] == 'long_spread':
                                self.oms_client.set_target_position(a_sym, instrument_type, 0.0, 'CLOSE')
                                self.oms_client.set_target_position(b_sym, instrument_type, 0.0, 'CLOSE')

                            # Short A, Long B * beta
                            self.oms_client.set_target_position(a_sym, instrument_type, a_usdt, 'SHORT')
                            self.oms_client.set_target_position(b_sym, instrument_type, b_usdt, 'LONG')
                            p['current_side'] = 'short_spread'
                        else:
                            if p['current_side'] == 'short_spread':
                                self.oms_client.set_target_position(a_sym, instrument_type, 0.0, 'CLOSE')
                                self.oms_client.set_target_position(b_sym, instrument_type, 0.0, 'CLOSE')
                            
                            # Long A, Short B * beta
                            self.oms_client.set_target_position(a_sym, instrument_type, a_usdt, 'LONG')
                            self.oms_client.set_target_position(b_sym, instrument_type, b_usdt, 'SHORT')
                            p['current_side'] = 'long_spread'

                        p['is_open'] = True
                        self._alloc_state[id(p)] = total_pair_usdt
                        self._last_entry_day[id(p)] = day_key
                        print(f"DEBUG opened side={p['current_side']} a_sym={a_sym} b_sym={b_sym} total_alloc={total_pair_usdt}")

            elif exit_ or stop:
                # Close both legs
                self.oms_client.set_target_position(a_sym, instrument_type, 0.0, 'CLOSE')
                self.oms_client.set_target_position(b_sym, instrument_type, 0.0, 'CLOSE')
                p['is_open'] = False
                p['current_side'] = None
                self._alloc_state[id(p)] = 0.0
                print(f"DEBUG closed pair=({a_base},{b_base})")
            else:
                # Hold
                continue


