from typing import List, Any, Dict
import numpy as np
import pandas as pd

# Module-level config setter to pass parameters from the runner script
_PAIRS_CONFIG: List[Dict] = []

def set_pairs_config(pairs_config: List[Dict]):
    global _PAIRS_CONFIG
    _PAIRS_CONFIG = pairs_config


class PairTradingStrategy:
    def __init__(self, symbols: List[str], oms_client: Any):
        self.symbols = symbols
        self.oms_client = oms_client
        # Build runtime state per pair
        self.pairs = []
        for cfg in _PAIRS_CONFIG:
            base_a, base_b = cfg['legs']
            self.pairs.append({
                'a_symbol': f"{base_a}-USDT-PERP" if cfg.get('use_futures', True) else f"{base_a}-USDT",
                'b_symbol': f"{base_b}-USDT-PERP" if cfg.get('use_futures', True) else f"{base_b}-USDT",
                'lookback_days': cfg.get('lookback_days', 7),
                'entry_z': cfg.get('entry_z', 1.0),
                'exit_z': cfg.get('exit_z', 0.5),
                'max_alloc_frac': cfg.get('max_alloc_frac', 0.2),
                'is_open': False,
                'current_side': None,  # 'long_spread' or 'short_spread'
                'last_beta': 1.0,
            })

    def _get_recent_closes(self, base_symbol: str, steps: int) -> np.ndarray:
        # Use mark OHLCV for futures; data_manager expects base symbol without -PERP
        dm = self.oms_client.data_manager
        ts = self.oms_client.current_time
        df = dm.get_data_at_time(base_symbol, ts, 'perpetual_mark_ohlcv')
        if df is None or df.empty:
            return np.array([])
        closes = df['close'].values
        if len(closes) < steps:
            return np.array([])
        return closes[-steps:]

    def _compute_beta_and_z(self, a_base: str, b_base: str, steps: int):
        a = self._get_recent_closes(a_base, steps)
        b = self._get_recent_closes(b_base, steps)
        if a.size == 0 or b.size == 0:
            return None, None, None
        # log-prices regression y = beta * x
        y = np.log(a)
        x = np.log(b)
        x_ = x.reshape(-1, 1)
        # Manual OLS for speed and no sklearn dependency
        x_mean = x_.mean()
        y_mean = y.mean()
        cov = ((x_ - x_mean) * (y - y_mean).reshape(-1, 1)).sum()
        var = ((x_ - x_mean) ** 2).sum()
        beta = (cov / var).item() if var != 0 else 1.0
        spread_series = y - beta * x
        curr_spread = spread_series[-1]
        mean_spread = spread_series.mean()
        std_spread = spread_series.std()
        z = (curr_spread - mean_spread) / std_spread if std_spread > 0 else 0.0
        return beta, z, std_spread

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

        for p in self.pairs:
            a_base = p['a_symbol'].replace('-PERP', '')
            b_base = p['b_symbol'].replace('-PERP', '')

            # 15m bars: 96 steps per day
            steps = max(int(p['lookback_days'] * 96), 96)
            beta, z, _ = self._compute_beta_and_z(a_base, b_base, steps)
            if beta is None:
                continue
            p['last_beta'] = beta

            a_sym = p['a_symbol']
            b_sym = p['b_symbol']

            instrument_type = 'future' if a_sym.endswith('-PERP') else 'spot'

            # Determine desired action
            enter = abs(z) > p['entry_z'] and not p['is_open']
            exit_ = abs(z) < p['exit_z'] and p['is_open']
            stop = abs(z) > (3.0 * p['entry_z']) and p['is_open']

            if enter:
                # Enforce one entry per calendar day
                now = self.oms_client.current_time
                day_key = (now.year, now.month, now.day)
                if self._last_entry_day.get(id(p)) == day_key:
                    # Already entered today; skip
                    pass
                else:
                    # Dollar allocation per pair
                    alloc = max(0.0, min(1.0, p['max_alloc_frac'])) * total_equity
                    # Scale with z-score up to 2x at z >= 2*entry
                    scale = min(abs(z) / max(p['entry_z'], 1e-6), 2.0)
                    leg_usdt = alloc * scale

                    # Check portfolio cap before opening
                    current_deployed = sum(self._alloc_state.values())
                    if current_deployed + leg_usdt <= max_alloc_total:
                        if z > 0:
                            # Short A, Long B * beta
                            self.oms_client.set_target_position(a_sym, instrument_type, leg_usdt, 'SHORT')
                            self.oms_client.set_target_position(b_sym, instrument_type, leg_usdt, 'LONG')
                            p['current_side'] = 'short_spread'
                        else:
                            # Long A, Short B * beta
                            self.oms_client.set_target_position(a_sym, instrument_type, leg_usdt, 'LONG')
                            self.oms_client.set_target_position(b_sym, instrument_type, leg_usdt, 'SHORT')
                            p['current_side'] = 'long_spread'

                        p['is_open'] = True
                        self._alloc_state[id(p)] = leg_usdt
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


