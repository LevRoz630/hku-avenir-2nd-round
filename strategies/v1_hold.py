from datetime import datetime
from typing import List
class HoldStrategy:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.current_time = datetime.now()
        self.has_bought = False

    def run_strategy(self):
        # Only buy once at the beginning, then hold
        if not self.has_bought:
            # Allocate all USDT equally to all symbols (assume all are spot or perpetual)
            balance = self.get_account_balance()
            usdt = balance.get("USDT", 0)
            if usdt > 0:
                num_symbols = len(self.symbols)
                if num_symbols == 0:
                    return
                allocation = (usdt - 1000) / num_symbols
                for symbol in self.symbols:
                    # Determine instrument type
                    self.set_target_position(
                        symbol=symbol,
                        instrument_type="spot",
                        target_value=allocation,
                        position_side="LONG"
                    )
                    self.set_target_position(
                        symbol=symbol,
                        instrument_type="future",
                        target_value=10,
                        position_side="LONG"
                    )
                self.has_bought = True
