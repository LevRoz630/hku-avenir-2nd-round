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


    def red_button(self, orders: List[Dict[str, Any]], oms_client: Any, data_manager: HistoricalDataCollector):
        """
        Red button to close positions due to n% loss.
        """
        self.oms_client = oms_client
        self.data_manager = data_manager
        
        current_positions = self.oms_client.get_position()

        for position in current_positions:
            if position['pnl'] < (0.05 * -position['entry_price']):
                logger.info(f"Closing position {position['symbol']} due to large loss of {position['pnl']}")
                orders.append({'symbol': position['symbol'], 'instrument_type': position['instrument_type'], 'side': 'CLOSE', 'value': 0.0})
    
    def prioritize_close_orders(self, orders: List[Dict[str, Any]]):
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
    
    def filter_orders(self, orders: List[Dict[str, Any]], oms_client: Any, data_manager: HistoricalDataCollector):
        self.oms_client = oms_client
        self.data_manager = data_manager

        orders = self.red_button(orders, oms_client, data_manager)
        orders = self.prioritize_close_orders(orders)

        if orders is None:
            return None
        
        return orders