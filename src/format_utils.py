"""
Formatting helpers for logging tables in CLI/console.
"""

from typing import Any, Dict, List


def format_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Return a simple ASCII table string.

    Args:
        headers: Column names
        rows: List of row lists
    """
    if not rows:
        return "(empty)"
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    header_line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(widths)))
    body_lines = [
        " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + body_lines)


def format_positions_table(positions: List[Dict[str, Any]]) -> str:
    """Format OMS positions list into an ASCII table."""
    rows: List[List[str]] = []
    for p in positions or []:
        inst = p.get('instrument_name') or p.get('symbol') or 'N/A'
        side = p.get('position_side', 'N/A')
        qty = p.get('quantity', '0')
        val = p.get('value', '0')
        avg = p.get('avg_price', p.get('entry_price', ''))
        try:
            qty = f"{float(qty):.6f}"
        except Exception:
            qty = str(qty)
        try:
            val = f"{float(val):.2f}"
        except Exception:
            val = str(val)
        try:
            avg = f"{float(avg):.2f}" if avg != '' else ''
        except Exception:
            avg = str(avg)
        rows.append([inst, side, qty, val, avg])
    return format_table(["Instrument", "Side", "Qty", "Value (USDT)", "Avg Price"], rows)


def format_balances_table(balances: Dict[str, Any]) -> str:
    """Format balances dict into an ASCII table."""
    rows: List[List[str]] = []
    for asset, bal in (balances or {}).items():
        try:
            bal_fmt = f"{float(bal):.2f}"
        except Exception:
            bal_fmt = str(bal)
        rows.append([asset, bal_fmt])
    rows.sort(key=lambda r: r[0])
    return format_table(["Asset", "Balance (USDT)"], rows)


def format_target_elements(elements: List[Dict[str, Any]]) -> str:
    """Format a list of target position instructions for logging."""
    rows: List[List[str]] = []
    for e in elements or []:
        rows.append([
            e.get("instrument_name", ""),
            e.get("instrument_type", ""),
            str(e.get("position_side", "")),
            str(e.get("target_value", "")),
        ])
    return format_table(["Instrument", "Type", "Side", "Target (USDT)"], rows)


def format_batch_result(result: Any) -> str:
    """Format batch API result (dict or list) into a table.

    Tries to extract common fields like id, instrument_name, type, side, value, times.
    """
    if result is None:
        return "(empty)"
    items: List[Dict[str, Any]]
    if isinstance(result, list):
        items = result
    elif isinstance(result, dict):
        # Some APIs return a single message dict
        items = [result]
    else:
        return str(result)

    rows: List[List[str]] = []
    for m in items:
        rows.append([
            str(m.get("id", "")),
            str(m.get("instrument_name", "")),
            str(m.get("instrument_type", "")),
            str(m.get("position_side", "")),
            str(m.get("target_value", "")),
            str(m.get("create_time", "")),
            str(m.get("update_time", "")),
        ])
    return format_table(["Task ID", "Instrument", "Type", "Side", "Target", "Created", "Updated"], rows)


