from datetime import datetime
from typing import List, Any
class HoldStrategy:
    def __init__(self, symbols: List[str], oms_client: Any):
        self.symbols = symbols
        self.oms_client = oms_client
        self.current_time = None
        self.has_bought = False

    def run_strategy(self):
        # Only buy once at the beginning, then hold
        if not self.has_bought:
            # Allocate all USDT equally to all symbols (assume all are spot or perpetual)
            balance = self.oms_client.get_account_balance()
            usdt = balance.get("USDT", 0)
            print(f"debug usdt:{usdt}")
            if usdt > 0:

                for symbol in self.symbols:
                    num_symbols = len(self.symbols)

                    allocation = (usdt - 1000) / num_symbols 
                    print(f"debug allocation:{allocation}")
                    # Determine instrument type
                    print(f"debug symbol:{symbol}")
                    self.oms_client.set_target_position(
                        symbol=symbol,
                        instrument_type="spot",
                        target_value=allocation,
                        position_side="LONG"
                    )
                    self.oms_client.set_target_position(
                        symbol=symbol,
                        instrument_type="future",
                        target_value=allocation,
                        position_side="LONG"
                    )
            self.has_bought = True
