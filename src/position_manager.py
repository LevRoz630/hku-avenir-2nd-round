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
        self.oms_client = oms_client
        self.data_manager = data_manager
        try:
            cleaned_orders = self._close_risky_orders(orders)
            weighted_orders = self._set_weights(cleaned_orders)
            print(f"DEBUG weighted_orders: {weighted_orders}")

            # this checks for the limit of current cash 
            if weighted_orders is not None:
                values = [x['value'] for x in weighted_orders]
                print(f"DEBUG values: {values}")
                print(f"DEBUG oms_client.balance['USDT']: {oms_client.balance['USDT']}")
                if sum(values) > oms_client.balance['USDT']:
                    logger.error(f"Insufficient USDT balance. Required: {values}, Available: {oms_client.balance['USDT']}")
                    return None
                return weighted_orders

            return None

        except Exception as e:
            logger.error(f"Error filtering orders: {e}")
            return None
        

    def _close_risky_orders(self, orders: List[Dict[str, Any]]):
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
                print(f"DEBUG scaled_vol {order['symbol']}: {scaled_vol}")
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
