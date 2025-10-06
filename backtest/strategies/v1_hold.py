from datetime import datetime
from typing import List, Any
from oms_simulation import OMSClient
from hist_data import HistoricalDataCollector

class HoldStrategy:
    def __init__(self, symbols: List[str], lookback_days: int):
        self.symbols = symbols
        self.oms_client = None
        self.data_manager = None
        self.has_bought = False
        self.orders = []
        self.lookback_days = lookback_days

    def run_strategy(self, oms_client: OMSClient, data_manager: HistoricalDataCollector):
        self.oms_client = oms_client
        self.data_manager = data_manager
        # Only buy once at the beginning, then hold
        if not self.has_bought:
            # Allocate all USDT equally to all symbols (assume all are spot or perpetual)
            usdt = self.oms_client.balance['USDT']
            if usdt > 0:
                for symbol in self.symbols:
                    # Determine instrument type
                    self.orders.append({
                        "symbol": symbol,
                        "instrument_type": "future",
                        "side": "LONG"
                    })
            self.has_bought = True
        return self.orders