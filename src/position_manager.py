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
        self.data_manager = HistoricalDataCollector()


    def filter_orders(self, orders: List[Dict[str, Any]], oms_client: Any):

        cleaned_orders = self._close_risky_orders(orders, oms_client)
        weighted_orders = self._set_weights(cleaned_orders, oms_client)

        return weighted_orders



    def _close_risky_orders(self, orders: List[Dict[str, Any]], oms_client: Any):
        for order in self.orders:
            try:
                data = self.data_manager.load_data_period(order['symbol'], '5m', 'mark_ohlcv_futures', self.oms_client.current_time - timedelta(hours = 2), self.oms_client.current_time)
                scaled_vol  =  np.std(data['close']) / np.mean(data['close'])
                if scaled_vol > 1.5:
                    order['value'] = 0
                logger.info(f"Closed risky order: {order['symbol']}")
            except Exception as e:
                logger.error(f"Error closing risky orders: {e} for: {order['symbol']}")

        return self.orders


    def _set_weights(self, orders: List[Dict[str, Any]], oms_client: Any):

            self.orders = orders
            self.oms_client = oms_client

            limit = self.oms_client.balance['USDT']
            for order in self.orders:
                try:
                    if order['value'] != 0:
                        data = self.data_manager.load_data_period(order['symbol'], '15m', 'mark_ohlcv_futures', self.oms_client.current_time - timedelta(days = 1), self.oms_client.current_time)
                        scaled_vol  =  np.std(data['close']) / np.mean(data['close'])
                        order_weight = len(self.orders) / scaled_vol
                        order['value'] = limit * order_weight
            
                except Exception as e:
                    logger.error(f"Error setting weights: {e} for {self.order['symbol']}")

            return self.orders
