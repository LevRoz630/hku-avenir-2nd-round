from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
from hist_data import HistoricalDataCollector
from oms_simulation import OMSClient

logger = logging.getLogger(__name__)

class V1LSPositionManager:
    def __init__(
        self,
        max_alloc_frac: float = 2000.0,
        ramp_up_days: int = 60,
        start_alloc_frac: float = 200.0,
    ):
        self.orders = []
        self.oms_client = None
        self.data_manager = None
        self.max_alloc_frac = max_alloc_frac
        self.ramp_up_days = ramp_up_days
        self.start_alloc_frac = start_alloc_frac
        self.start_time: Optional[datetime] = None

    def _set_oms_and_dm(self, oms_client: Any, data_manager: HistoricalDataCollector) -> None:
        self.oms_client = oms_client
        self.data_manager = data_manager
        # Track start time on first call for ramp-up calculation
        if self.start_time is None and oms_client.current_time is not None:
            self.start_time = oms_client.current_time

    def _red_button(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Close positions with >15% unrealized loss."""
        try:
            current_positions = self.oms_client.get_position() or []
            for position in current_positions:
                # Coerce string fields from OMS to floats for safe arithmetic
                try:
                    qty = float(position.get('quantity', 0.0))
                    entry_price = float(position.get('entry_price', 0.0))
                    pnl = float(position.get('pnl', 0.0))
                except (TypeError, ValueError):
                    continue

                # Calculate entry value (always positive)
                entry_value = abs(qty) * entry_price
                if entry_value == 0:
                    continue
                
                # Calculate PnL percentage
                pnl_pct = pnl / entry_value
                
                # Close if we lost more than 15% (works for both LONG and SHORT)
                if pnl_pct < -0.15:
                    logger.info(f"Closing position {position['symbol']} due to large loss of {pnl:.2f} ({pnl_pct*100:.2f}%)")
                    orders.append({'symbol': position['symbol'], 'instrument_type': position['instrument_type'], 'side': 'CLOSE'})
            return orders
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


    def _get_current_alloc_multiplier(self) -> float:
        """
        Calculate current allocation multiplier based on elapsed time.
        Ramps linearly from start_alloc_frac to max_alloc_frac over ramp_up_days.
        """
        if self.start_time is None or self.oms_client is None or self.oms_client.current_time is None:
            return self.start_alloc_frac
        
        elapsed = self.oms_client.current_time - self.start_time
        elapsed_days = elapsed.total_seconds() / 86400.0
        
        if elapsed_days <= 0:
            return self.start_alloc_frac
        
        if elapsed_days >= self.ramp_up_days:
            return self.max_alloc_frac
        
        # Linear interpolation
        progress = elapsed_days / self.ramp_up_days
        current_multiplier = self.start_alloc_frac + (
            (self.max_alloc_frac - self.start_alloc_frac) * progress
        )
        
        return current_multiplier

    def _set_weights(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not orders:
            return []
        
        current_multiplier = self._get_current_alloc_multiplier()
        sized: List[Dict[str, Any]] = []
        
        for order in orders:
            side = order.get('side')
            # CLOSE orders don't need alloc_frac/value; pass through
            if side == 'CLOSE':
                sized.append(order)
                continue
            alloc = order.get('alloc_frac', 0.0)
            try:
                order['value'] = float(current_multiplier) * float(alloc)
            except Exception:
                order['value'] = 0.0
            sized.append(order)
        return sized
    
    def filter_orders(self, orders: List[Dict[str, Any]], oms_client: OMSClient, data_manager: HistoricalDataCollector) -> List[Dict[str, Any]]:

        try:
            self._set_oms_and_dm(oms_client, data_manager)
            incoming = orders or []
            after_rb = self._red_button(incoming)
            after_weights = self._set_weights(after_rb)
            prioritized = self._prioritize_close_orders(after_weights)
            return prioritized or []
        except Exception as e:
            logger.error(f"Error filtering orders: {e}")
            return []