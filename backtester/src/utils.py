from datetime import datetime, timezone
import pandas as pd

# Helper methods with documentation
def _convert_symbol_to_ccxt(symbol: str, market_type: str = "spot"):
    """Convert symbol to CCXT format (spot: BTC/USDT, futures: BTC/USDT:USDT)."""
    try:
        if market_type == "spot":
            # For spot: BTC-USDT -> BTC/USDT
            if '-' in symbol:
                return symbol.replace('-', '/')
            return symbol
        else:
            # For futures: BTC-USDT-PERP -> BTC/USDT:USDT
            base = symbol.split('-')[0]
            return f"{base}/USDT:USDT"
    except:
        return None

def _normalize_symbol_pair(symbol: str) -> str | None:
    """
    Normalize various symbol inputs to CCXT pair format BASE/QUOTE.
    Examples:
        - "BTC-USDT-PERP" -> "BTC/USDT"
        - "BTC-USDT" -> "BTC/USDT"
        - "BTC/USDT" -> "BTC/USDT" (unchanged)
    """
    try:
        s = str(symbol).strip()
        if not s:
            return None
        # Already in CCXT pair format
        if '/' in s:
            base, quote = s.split('/', 1)
            return f"{base.upper()}/{quote.upper()}"
        # Remove potential "-PERP" suffix and split by '-'
        s = s.upper().replace("-PERP", "")
        parts = [p for p in s.split('-') if p]
        if len(parts) == 1:
            base = parts[0]
            quote = 'USDT'
        else:
            base, quote = parts[0], parts[1]
        return f"{base}/{quote}"
    except Exception:
        return None
            

def _is_utc(dt: datetime) -> None:
    if dt.tzinfo is timezone.utc:
        return
    else:
        raise ValueError("Timestamp is not in UTC timezone, please use datetime.now(timezone.utc) or similar")


def _get_number_of_periods(timeframe: str, start_time: datetime, end_time: datetime):
    minutes = _get_timeframe_to_minutes(timeframe)
    total_minutes = (end_time - start_time).total_seconds() / 60
    total_periods = int(total_minutes // minutes)
    return total_periods

def _get_timeframe_to_minutes(timeframe: str):
    periods_map = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360,
        '12h': 720, '1d': 1440,
    }
    return periods_map.get(timeframe, 15)
