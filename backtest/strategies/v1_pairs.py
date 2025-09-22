from typing import List, Any, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
# Module-level config setter to pass parameters from the runner script
_PAIRS_CONFIG: List[Dict] = []

def set_pairs_config(pairs_config: List[Dict]):
    global _PAIRS_CONFIG
    _PAIRS_CONFIG = pairs_config


class PairTradingStrategy:
    def __init__(self, symbols: List[str], oms_client: Any, steps: int):
        self.symbols = symbols
        self.oms_client = oms_client
        self.steps = steps
        # Build runtime state per pair
        self.pairs = []
        for cfg in _PAIRS_CONFIG:
            base_a, base_b = cfg['legs']
            self.pairs.append({
                'a_symbol': f"{base_a}" if cfg.get('use_futures', True) else f"{base_a}",
                'b_symbol': f"{base_b}" if cfg.get('use_futures', True) else f"{base_b}",
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
        dm = self.oms_client.data_manager
        ts = self.oms_client.current_time
        # Use existing collector method which returns df if parquet exists, else collects then returns
        df = dm.collect_perpetual_mark_ohlcv(base_symbol, timeframe='15m', days=40)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        df = df.copy()
        df = df.sort_values('timestamp')
        df = df.set_index('timestamp')
        # Resample to daily closes
        daily = df['close'].resample('1D').last().dropna()
        # Align to prior day close relative to current time
        daily = daily[daily.index.date < self.oms_client.current_time.date()]
        return daily

    def _compute_beta_and_z(self, a_base: str, b_base: str, lookback_days: int):
        """Compute dynamic beta on daily log-prices and z-score like offline.

        - Beta from regression of log(y) on log(x) over last `lookback_days` daily closes (excluding current day)
        - Current spread and historical spreads computed on raw prices using that beta
        - z = (current_spread - mean(historical_spreads)) / std(historical_spreads)
        """
        a_series = self._get_daily_closes(a_base)
        print(f"debug a_series:{a_series}")
        b_series = self._get_daily_closes(b_base)
        print(f"debug b_series:{b_series}")
        if a_series.empty or b_series.empty:
            return None, None, None
        # Align on common dates
        df = pd.concat([a_series.rename('a'), b_series.rename('b')], axis=1).dropna()
        if len(df) < lookback_days + 1:
            return None, None, None
        # Use last (lookback_days + 1) daily points; last one is "current" bar
        window = df.iloc[-(lookback_days + 1):]
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
        return beta, float(z), spread_std


    def run_strategy(self):
        # Portfolio value for sizing
        total_equity = float(self.oms_client.get_total_portfolio_value() or 0.0)

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
        is_decision_time = (now.hour == 0 and now.minute == 0)

        for p in self.pairs:
            a_base = p['a_symbol'].replace('-PERP', '')
            b_base = p['b_symbol'].replace('-PERP', '')

            # Compute daily beta/z only at decision time; otherwise hold
            if not is_decision_time:
                continue

            beta, z, std = self._compute_beta_and_z(a_base, b_base, p['lookback_days'])
            if beta is None:
                continue
            p['last_beta'] = beta

            a_sym = p['a_symbol']
            b_sym = p['b_symbol']

            instrument_type = 'future' if a_sym.endswith('-PERP') else 'spot'

            # Determine desired action
            enter = abs(z) > p['entry_z']
            exit_ = abs(z) < p['exit_z'] and p['is_open']
            stop = abs(z) > (3.0 * p['entry_z']) and p['is_open']

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

            elif exit_ or stop:
                # Close both legs
                self.oms_client.set_target_position(a_sym, instrument_type, 0.0, 'CLOSE')
                self.oms_client.set_target_position(b_sym, instrument_type, 0.0, 'CLOSE')
                p['is_open'] = False
                p['current_side'] = None
                self._alloc_state[id(p)] = 0.0
            else:
                # Hold
                continue


