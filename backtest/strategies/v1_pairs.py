from typing import List, Any, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta, timezone
from hist_data import HistoricalDataCollector
# Module-level config setter to pass parameters from the runner script
_PAIRS_CONFIG: List[Dict] = []

def set_pairs_config(pairs_config: List[Dict]):
    global _PAIRS_CONFIG
    _PAIRS_CONFIG = pairs_config


class PairTradingStrategy:
    def __init__(self, symbols: List[str], historical_data_dir: str, lookback_days: int):
        self.symbols = symbols
        self.current_time = None
        self.lookback_days = lookback_days 
        # Build runtime state per pair
        self.pairs = []
        self.historical_data = pd.DataFrame()
        self.current_time = None
        self.end_time = None
        self.orders = []

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

    def _get_daily_closes(self, base_symbol: str) -> pd.Series:
        """Return daily close series up to current time using mark OHLCV.

        Falls back to resampling 15m/1h data to daily closes if 1d is not present.
        """
        dm = self.data_manager
        # Use existing collector method which returns the past n days, set export to False as we don't want to use saved data over and over again

        # This conditional should reduce the time we spend collecting data as we only collect full forty days at the start and then check if we need to collect more
        
        # Fetch a rolling window to ensure we have prior days before current day
        window_days = max(self.lookback_days + 2, 3)
        window_start = self.current_time - pd.Timedelta(days=window_days)
        end_for_load = self.current_time
        df = dm.load_data_period(base_symbol, timeframe='15m', data_type='index_ohlcv_futures', start_date=window_start, end_date=end_for_load)

        df = df[df['timestamp'].between(window_start, self.current_time, inclusive='left')]

        df = df.sort_values('timestamp')
        df = df.set_index('timestamp')
        # Resample to daily closes
        daily = df['close'].resample('1D').last().dropna()

        daily = daily.iloc[-self.lookback_days:]
        # Don't take the last close as it might leak future info about the current day 
        cutoff = self.current_time - pd.Timedelta(days=1) 
        daily = daily[daily.index < cutoff]
        print(f"DEBUG filtered_pre_now days={len(daily)} last_idx={daily.index.max()} cutoff={cutoff}, shape:{daily.shape}")
        return daily

    def _compute_beta_and_z(self, a_base: str, b_base: str):
        """Compute dynamic beta on daily log-prices and z-score like offline.

        - Beta from regression of log(y) on log(x) over last `lookback_days` daily closes (excluding current day)
        - Current spread and historical spreads computed on raw prices using that beta
        - z = (current_spread - mean(historical_spreads)) / std(historical_spreads)
        """

        a_series = self._get_daily_closes(a_base)
        b_series = self._get_daily_closes(b_base)
        print(f"debug a series shape{a_series.shape} b series shape{b_series.shape}")
        if a_series.empty or b_series.empty:
            print(f"DEBUG series_empty a_len={len(a_series)} b_len={len(b_series)}")
            return None, None, None
        # Align on common dates
        df = pd.concat([a_series.rename('a'), b_series.rename('b')], axis=1).dropna()
        print(f"DEBUG aligned_len={len(df)} a_range=({a_series.index.min()} -> {a_series.index.max()}) b_range=({b_series.index.min()} -> {b_series.index.max()})")
        # Use last (lookback + 1) daily points; last one is "current" bar
        window = df.iloc[-(self.lookback_days + 1):]
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


    def run_strategy(self, current_time: datetime, data_manager: HistoricalDataCollector):
        
        # Track last entry day per pair to enforce one entry per day
        if not hasattr(self, '_last_entry_day'):
            self._last_entry_day = {id(p): None for p in self.pairs}

        # Only make entry/exit decisions once per day using daily signals
        self.current_time = current_time    
        self.data_manager = data_manager

        for p in self.pairs:
            a_base = p['a_symbol'].replace('-PERP', '')
            b_base = p['b_symbol'].replace('-PERP', '')

            print(f"DEBUG compute_params pair=({a_base},{b_base}) lookback={self.lookback_days}")
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

            print(f"DEBUG decision t={self.current_time} pair=({a_base},{b_base}) z={z} enter={abs(z) > p['entry_z']} exit={abs(z) < p['exit_z'] and p['is_open']} stop={abs(z) > (3.0 * p['entry_z'])}")
            if enter:
                # Enforce one entry per calendar day
                day_key = (self.current_time.year, self.current_time.month, self.current_time.day)
                if self._last_entry_day.get(id(p)) == day_key:
                    # Already entered today; skip
                    pass
                else:
                  
                    if z > 0:
                        if p['current_side'] == 'long_spread':
                            self.orders.append({'symbol': a_sym, 'instrument_type': instrument_type, 'side': 'CLOSE'})
                            self.orders.append({'symbol': b_sym, 'instrument_type': instrument_type, 'side': 'CLOSE'})

                        # Short A, Long B * beta
                        self.orders.append({'symbol': a_sym, 'instrument_type': instrument_type, 'side': 'SHORT'})
                        self.orders.append({'symbol': b_sym, 'instrument_type': instrument_type, 'side': 'LONG'})
                        p['current_side'] = 'short_spread'
                    else:
                        if p['current_side'] == 'short_spread':
                            self.orders.append({'symbol': a_sym, 'instrument_type': instrument_type, 'side': 'CLOSE'})
                            self.orders.append({'symbol': b_sym, 'instrument_type': instrument_type, 'side': 'CLOSE'})
                        
                        # Long A, Short B * beta
                        self.orders.append({'symbol': a_sym, 'instrument_type': instrument_type, 'side': 'LONG'})
                        self.orders.append({'symbol': b_sym, 'instrument_type': instrument_type, 'side': 'SHORT'})
                        p['current_side'] = 'long_spread'

                        p['is_open'] = True
                        self._last_entry_day[id(p)] = day_key

            elif exit_ or stop:
                # Close both legs
                self.orders.append({'symbol': a_sym, 'instrument_type': instrument_type, 'side': 'CLOSE'})
                self.orders.append({'symbol': b_sym, 'instrument_type': instrument_type, 'side': 'CLOSE'})
                p['is_open'] = False
                p['current_side'] = None
                
            else:
                # Hold
                continue


