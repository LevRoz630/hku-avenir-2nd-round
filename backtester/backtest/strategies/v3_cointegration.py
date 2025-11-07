"""Basket cointegration trading strategy (v3).

Trades multi-asset baskets using eigenvector weights derived from the
cointegration hypothesis testing pipeline. Uses optimized z-score
thresholds to generate entry/exit signals and emits OMS orders in the
format expected by the position manager.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hist_data import HistoricalDataCollector  # type: ignore
from oms_simulation import OMSClient  # type: ignore

TIMEFRAME_TO_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "12h": 720,
    "1d": 1440,
}


@dataclass
class BasketState:
    """Runtime state for each basket."""

    is_open: bool = False
    side: Optional[str] = None  # "long_spread" or "short_spread"
    last_entry_time: Optional[datetime] = None
    last_entry_z: Optional[float] = None
    last_zscore: Optional[float] = None


class V3CointegrationStrategy:
    """Execute cointegration trades on pre-validated baskets."""

    def __init__(
        self,
        baskets: List[Dict],
        *,
        historical_data_dir: str,
        timeframe: str = "15m",
        stop_multiplier: float = 3.0,
        min_history_buffer_days: int = 2,
    ) -> None:
        self.basket_configs: List[Dict] = []
        self.basket_states: List[BasketState] = []

        minutes = TIMEFRAME_TO_MINUTES.get(timeframe, 15)
        self.timeframe = timeframe
        self.timeframe_minutes = minutes
        self.timeframe_delta = timedelta(minutes=minutes)
        self.bars_per_day = max(int(1440 / minutes), 1)
        self.stop_multiplier = stop_multiplier
        self.min_history_buffer_days = max(min_history_buffer_days, 1)

        self.historical_data_dir = historical_data_dir
        self.data_manager: Optional[HistoricalDataCollector] = None
        self.oms_client: Optional[OMSClient] = None

        # Collect union of symbols for data pre-loading
        symbols_usdt: List[str] = []
        max_lookback_days = 0

        for basket in baskets:
            eigenvector = np.array(basket.get("eigenvector", []), dtype=float)
            if eigenvector.size == 0:
                continue

            sum_abs = np.sum(np.abs(eigenvector))
            if not np.isfinite(sum_abs) or sum_abs == 0:
                continue

            lookback_days = int(basket.get("lookback_days", 30))
            max_lookback_days = max(max_lookback_days, lookback_days)

            config = {
                **basket,
                "eigenvector": eigenvector,
                "sum_abs_weights": float(sum_abs),
                "lookback_days": lookback_days,
                "lookback_bars": max(int(lookback_days * self.bars_per_day), 1),
            }

            self.basket_configs.append(config)
            self.basket_states.append(BasketState())

            for sym in basket.get("symbols_usdt", []):
                if sym not in symbols_usdt:
                    symbols_usdt.append(sym)

        if not self.basket_configs:
            raise ValueError("No valid basket configurations provided")

        self.symbols = symbols_usdt
        self.lookback_days = max_lookback_days

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_oms_and_dm(
        self, oms_client: OMSClient, data_manager: HistoricalDataCollector
    ) -> None:
        self.oms_client = oms_client
        self.data_manager = data_manager

    def _load_basket_price_history(self, basket_cfg: Dict) -> Optional[pd.DataFrame]:
        assert self.oms_client is not None
        assert self.data_manager is not None

        end_time = self.oms_client.current_time - self.timeframe_delta
        start_time = end_time - timedelta(
            days=basket_cfg["lookback_days"] + self.min_history_buffer_days
        )

        frames = []
        for symbol, symbol_usdt in zip(
            basket_cfg.get("symbols", []), basket_cfg.get("symbols_usdt", [])
        ):
            df = self.data_manager.load_data_period(
                symbol_usdt,
                self.timeframe,
                "mark_ohlcv_futures",
                start_time,
                end_time,
                load_from_class=True,
            )

            if df is None or df.empty:
                return None

            series = (
                df.sort_values("timestamp")
                .set_index("timestamp")["close"]
                .rename(symbol)
                .astype(float)
            )
            frames.append(series)

        price_df = pd.concat(frames, axis=1).dropna()

        if price_df.empty:
            return None

        # Ensure chronological order and deduplicated timestamps
        price_df = price_df[~price_df.index.duplicated(keep="last")]
        price_df = price_df.sort_index()
        return price_df

    @staticmethod
    def _compute_zscore(spread: pd.Series, lookback_bars: int) -> Optional[float]:
        if spread is None or spread.empty:
            return None

        if len(spread) <= lookback_bars:
            return None

        window = spread.iloc[-lookback_bars:]
        mean = window.mean()
        std = window.std(ddof=1)
        if std is None or std <= 0:
            return None

        current = spread.iloc[-1]
        return float((current - mean) / std)

    def _build_basket_orders(
        self, basket_cfg: Dict, side: str
    ) -> List[Dict[str, object]]:
        assert side in {"long_spread", "short_spread"}

        orders: List[Dict[str, object]] = []
        basket_id = basket_cfg.get("name", f"basket_{id(basket_cfg)}")
        
        # Convert eigenvector weights to ratios (normalized absolute values)
        eigenvector = basket_cfg["eigenvector"]
        abs_weights = np.abs(eigenvector)
        sum_abs = np.sum(abs_weights)
        
        if sum_abs <= 0:
            return orders

        for symbol_perp, weight in zip(
            basket_cfg.get("symbols_perp", []), eigenvector
        ):
            if weight == 0:
                continue

            # Use absolute normalized weight as ratio for v3_pairs_pm
            ratio = abs(weight) / sum_abs if sum_abs > 0 else 0
            
            if ratio <= 0:
                continue

            if side == "long_spread":
                order_side = "LONG" if weight > 0 else "SHORT"
            else:
                order_side = "SHORT" if weight > 0 else "LONG"

            orders.append(
                {
                    "symbol": symbol_perp,
                    "instrument_type": "future",
                    "side": order_side,
                    "pair_id": basket_id,
                    "ratio": float(ratio),
                    "eigenvector_sign": float(np.sign(weight)),  # Store original sign for spread calculation
                }
            )

        return orders

    def _close_basket_orders(self, basket_cfg: Dict) -> List[Dict[str, object]]:
        orders: List[Dict[str, object]] = []
        basket_id = basket_cfg.get("name", f"basket_{id(basket_cfg)}")
        for symbol_perp in basket_cfg.get("symbols_perp", []):
            orders.append(
                {
                    "symbol": symbol_perp,
                    "instrument_type": "future",
                    "side": "CLOSE",
                    "pair_id": basket_id,
                }
            )
        return orders

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_strategy(
        self, oms_client: OMSClient, data_manager: HistoricalDataCollector
    ) -> List[Dict[str, object]]:
        self._set_oms_and_dm(oms_client, data_manager)

        orders: List[Dict[str, object]] = []

        for idx, (basket_cfg, state) in enumerate(
            zip(self.basket_configs, self.basket_states)
        ):
            price_df = self._load_basket_price_history(basket_cfg)
            if price_df is None:
                continue

            log_prices = np.log(price_df)
            spread = log_prices.values @ basket_cfg["eigenvector"]
            spread_series = pd.Series(spread, index=price_df.index)

            zscore = self._compute_zscore(spread_series, basket_cfg["lookback_bars"])
            if zscore is None:
                continue

            state.last_zscore = zscore

            entry_threshold = float(basket_cfg.get("entry_threshold", 1.5))
            exit_threshold = float(basket_cfg.get("exit_threshold", 0.5))
            stop_threshold = self.stop_multiplier * entry_threshold

            # No position open -> evaluate entry
            if not state.is_open:
                if zscore >= entry_threshold:
                    entry_orders = self._build_basket_orders(
                        basket_cfg, "short_spread"
                    )
                    if entry_orders:
                        orders.extend(entry_orders)
                        state.is_open = True
                        state.side = "short_spread"
                        state.last_entry_time = self.oms_client.current_time
                        state.last_entry_z = zscore
                elif zscore <= -entry_threshold:
                    entry_orders = self._build_basket_orders(
                        basket_cfg, "long_spread"
                    )
                    if entry_orders:
                        orders.extend(entry_orders)
                        state.is_open = True
                        state.side = "long_spread"
                        state.last_entry_time = self.oms_client.current_time
                        state.last_entry_z = zscore
                continue

            # Position already open -> manage risk/exit/flip
            if abs(zscore) <= exit_threshold:
                orders.extend(self._close_basket_orders(basket_cfg))
                state.is_open = False
                state.side = None
                state.last_entry_time = None
                state.last_entry_z = None
                continue

            if abs(zscore) >= stop_threshold:
                orders.extend(self._close_basket_orders(basket_cfg))
                state.is_open = False
                state.side = None
                state.last_entry_time = None
                state.last_entry_z = None
                continue

            if state.side == "long_spread" and zscore >= entry_threshold:
                orders.extend(self._close_basket_orders(basket_cfg))
                state.is_open = False
                state.side = None
                state.last_entry_time = None
                state.last_entry_z = None

                entry_orders = self._build_basket_orders(basket_cfg, "short_spread")
                if entry_orders:
                    orders.extend(entry_orders)
                    state.is_open = True
                    state.side = "short_spread"
                    state.last_entry_time = self.oms_client.current_time
                    state.last_entry_z = zscore
                continue

            if state.side == "short_spread" and zscore <= -entry_threshold:
                orders.extend(self._close_basket_orders(basket_cfg))
                state.is_open = False
                state.side = None
                state.last_entry_time = None
                state.last_entry_z = None

                entry_orders = self._build_basket_orders(basket_cfg, "long_spread")
                if entry_orders:
                    orders.extend(entry_orders)
                    state.is_open = True
                    state.side = "long_spread"
                    state.last_entry_time = self.oms_client.current_time
                    state.last_entry_z = zscore

        return orders


__all__ = ["V3CointegrationStrategy"]


