import os 
from re import L
from typing import List, Dict, Any
import logging
from datetime import timedelta, timezone, datetime
from hist_data import HistoricalDataCollector
import numpy as np

logger = logging.getLogger(__name__)

class PositionManager:
    def __init__(self):
        self.orders = []
        self.oms_client = None
        self.data_manager = None


    def filter_orders(self, orders: List[Dict[str, Any]], oms_client: Any, data_manager: HistoricalDataCollector):
        """
        Pipeline to validate and weight raw strategy orders before sending to OMS.

        Steps:
        1) Risk screen via short-horizon volatility; orders over threshold get value=0.
        2) Size remaining orders using inverse-vol weights under a USDT budget.
        3) Enforce balance constraint; if sized orders exceed cash, reject the batch.

        Returns a list of enriched order dicts or None if rejected.
        """
        self.oms_client = oms_client
        self.data_manager = data_manager
        try:
            # Separate CLOSE orders to bypass sizing/budget gating
            close_orders = [o for o in orders if o.get('side') == 'CLOSE']
            open_orders = [o for o in orders if o.get('side') != 'CLOSE']

            cleaned_open = self._close_risky_orders(open_orders)
            weighted_open = self._set_weights(cleaned_open)

            # Enforce balance constraint only on opens
            if weighted_open is not None:
                values = [x.get('value', 0.0) for x in weighted_open]
                if sum(values) > oms_client.balance['USDT']:
                    logger.error(f"Insufficient USDT balance. Required: {values}, Available: {oms_client.balance['USDT']}")
                    return close_orders if close_orders else None
                return close_orders + weighted_open

            # If no open orders after filtering, return CLOSE orders if any
            return close_orders if close_orders else None

        except Exception as e:
            logger.error(f"Error filtering orders: {e}")
            return None
        

    def _close_risky_orders(self, orders: List[Dict[str, Any]]):
        """
        Mark orders as value=0 when recent realized vol is above a threshold.

        - Uses 4 hours of 15m mark OHLCV to compute scaled volatility
          (std/mean). Current threshold: 0.1.
        - Futures data is stored under base symbols; '-PERP' suffix is removed.
        """
        cleaned: List[Dict[str, Any]] = []
        for order in orders:
            try:
                base_symbol = order['symbol'].replace('-PERP', '')
                data = self.data_manager.load_data_period(
                    base_symbol,
                    '15m',
                    'mark_ohlcv_futures',
                    self.oms_client.current_time - timedelta(hours=4),
                    self.oms_client.current_time
                )
                if data is None or len(data) == 0:
                    cleaned.append(order)
                    continue
                scaled_vol = float(np.std(data['close']) / np.mean(data['close']))
                if scaled_vol > 0.1:
                    # Flag as do-not-trade by setting value 0
                    new_order = {**order, 'value': 0}
                    cleaned.append(new_order)
                else:
                    cleaned.append(order)
            except Exception as e:
                logger.error(f"Error closing risky orders: {e} for: {order['symbol']}")
                cleaned.append(order)

        return cleaned


    def _set_weights(self, orders: List[Dict[str, Any]]):
            """
            Size orders by inverse volatility under a budget cap.

            - Budget: 10% of current USDT balance (conservative sizing).
            - For each non-zero order, compute 1/scaled_vol over last 1 day of 15m data.
            - Allocate proportionally to inverse-vol weights.
            """
            # Work on a copy to avoid mutating the input list unexpectedly
            updated: List[Dict[str, Any]] = []
            limit = self.oms_client.balance['USDT'] / 10

            # Compute inverse-vol weights for non-zero orders
            inv_vols = []
            for order in orders:
                if order.get('value', None) == 0:
                    inv_vols.append(0.0)
                    continue
                try:
                    base_symbol = order['symbol'].replace('-PERP', '')
                    data = self.data_manager.load_data_period(
                        base_symbol,
                        '15m',
                        'mark_ohlcv_futures',
                        self.oms_client.current_time - timedelta(days=1),
                        self.oms_client.current_time
                    )
                    if data is None or len(data) == 0:
                        inv_vols.append(0.0)
                        continue
                    scaled_vol = float(np.std(data['close']) / np.mean(data['close']))
                    inv_vols.append(0.0 if scaled_vol <= 0 else 1.0 / scaled_vol)
                except Exception:
                    inv_vols.append(0.0)

            total_inv = sum(inv_vols) if inv_vols else 0.0

            for idx, order in enumerate(orders):
                try:
                    if order.get('value', None) == 0:
                        updated.append(order)
                        continue
                    weight = (inv_vols[idx] / total_inv) if total_inv > 0 else 0.0
                    updated.append({**order, 'value': limit * weight})
                except Exception as e:
                    logger.error(f"Error setting weights: {e} for {order['symbol']}")
                    updated.append(order)

            return updated
