import os 
from re import L
from typing import List, Dict, Any
import logging
from datetime import timedelta, timezone, datetime
from hist_data import HistoricalDataCollector
from oms_simulation import OMSClient
import numpy as np

logger = logging.getLogger(__name__)

class V1LSPositionManager:
    def __init__(self):
        self.orders = []
        self.oms_client = None
        self.data_manager = None
        self.max_alloc_frac = 2000

    def _set_oms_and_dm(self, oms_client: Any, data_manager: HistoricalDataCollector) -> None:
        self.oms_client = oms_client
        self.data_manager = data_manager

    def _red_button(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Red button to close positions due to n% loss.
        """
        try:
            current_positions = self.oms_client.get_position()

            for position in current_positions:
                # checks if we lost more than 5% on a position (short or long)
                if  position['value'] < 0.95*position['entry_price']*position['quantity']:
                    logger.info(f"Closing position {position['symbol']} due to large loss of {position['pnl']}")
                    orders.append({'symbol': position['symbol'], 'instrument_type': position['instrument_type'], 'side': 'CLOSE'})
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return orders
        

    def _prioritize_close_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            order['value'] = self.max_alloc_frac * order['alloc_frac']
            
    
    def filter_orders(self, orders: List[Dict[str, Any]], oms_client: OMSClient, data_manager: HistoricalDataCollector) -> List[Dict[str, Any]]:

        try:
            self._set_oms_and_dm(oms_client, data_manager)
            orders = self._red_button(orders)
            orders = self._set_weights(orders)
            orders = self._prioritize_close_orders(orders)

            if orders is None:
                return []
        except Exception as e:
            logger.error(f"Error filtering orders: {e}")
            return []
        
        return orders