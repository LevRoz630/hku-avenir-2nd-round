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
        self.max_alloc_frac = 2000

    def oms_and_dm(self, oms_client: Any, data_manager: HistoricalDataCollector) -> None:
        self.oms_client = oms_client
        self.data_manager = data_manager

    def red_button(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Red button to close positions due to n% loss.
        """
        current_positions = self.oms_client.get_position()

        for position in current_positions:
            # close if the coin drops by more than 5% since we bought
            if position['pnl'] < (0.05 * -position['entry_price']):
                logger.info(f"Closing position {position['symbol']} due to large loss of {position['pnl']}")
                orders.append({'symbol': position['symbol'], 'instrument_type': position['instrument_type'], 'side': 'CLOSE', 'value': 0.0})
        

    def prioritize_close_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_symbol = {}
        for o in orders:
            sym = o.get('symbol')
            if not sym:
                continue
            if o.get('side') == 'CLOSE':
                by_symbol[sym] = o
            elif sym not in by_symbol or by_symbol[sym].get('side') != 'CLOSE':
                by_symbol[sym] = o
        return list(by_symbol.values())


    def _set_weights(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for order in orders:
            
    
    def filter_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        orders = self.red_button(orders)
        orders = self._set_weights(orders)
        orders = self.prioritize_close_orders(orders)


        if orders is None:
            return []
        
        return orders