"""Utilities for loading validated cointegration basket configs.

Combines basket eigenvectors from the validation pipeline with
optimized trading parameters so strategies can consume a single
configuration object per basket.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def _normalize_basket_key(basket: Sequence[str]) -> tuple[str, ...]:
    """Normalize basket key for dictionary lookups.

    Uses tuple with the original ordering to preserve hedge ratio mapping.
    """

    return tuple(str(symbol).upper() for symbol in basket)


def load_cointegration_basket_configs(
    validated_path: str | Path,
    optimization_path: str | Path,
    *,
    default_max_alloc_frac: float = 1.0,
) -> List[Dict]:
    """Load validated baskets and merge in z-score optimization results.

    Args:
        validated_path: Path to ``validated_baskets.json`` produced by the
            hypothesis testing pipeline.
        optimization_path: Path to ``zscore_optimization.json`` containing
            per-basket optimal parameters.
        default_max_alloc_frac: Optional max allocation fraction used by the
            live strategy when sizing orders. This gets applied when the JSON
            files do not specify a custom value.

    Returns:
        List of basket configuration dictionaries sorted in the same order as
        the validated baskets file. Each entry contains:

        ``symbols``: Raw symbols (e.g. ``"GRT"``)
        ``symbols_usdt``: Symbols formatted for historical data collector
        ``symbols_perp``: Symbols formatted for OMS trading
        ``eigenvector``: Eigenvector weights from the Johansen test
        ``entry_threshold`` / ``exit_threshold`` / ``lookback_days``: Optimized
        parameters from the z-score search
        ``metadata``: Dict with diagnostic metrics (p-values, half-life, etc.)
        ``max_alloc_frac``: Allocation budget for the basket
    """

    validated_path = Path(validated_path)
    optimization_path = Path(optimization_path)

    with validated_path.open("r") as fh:
        validated_payload = json.load(fh)

    with optimization_path.open("r") as fh:
        optimization_payload = json.load(fh)

    validated_config = validated_payload.get("config", {})
    baskets: Iterable[Dict] = validated_payload.get("baskets", [])

    optimization_results: Iterable[Dict] = optimization_payload.get(
        "optimization_results", []
    )

    optimization_lookup: Dict[tuple[str, ...], Dict] = {}
    for result in optimization_results:
        basket_symbols = result.get("basket", [])
        key = _normalize_basket_key(basket_symbols)
        optimization_lookup[key] = result

    combined_configs: List[Dict] = []
    timeframe = validated_config.get("timeframe", "15m")
    price_type = validated_config.get("price_type", "mark")

    for idx, basket_record in enumerate(baskets):
        basket_symbols = basket_record.get("basket", [])
        key = _normalize_basket_key(basket_symbols)

        optimization_entry = optimization_lookup.get(key)
        if not optimization_entry:
            # Skip baskets without optimization results – prevents accidental
            # trading of unoptimized combos.
            continue

        optimal_params: Optional[Dict] = optimization_entry.get("optimal_params")
        if not optimal_params:
            continue

        eigenvector: Sequence[float] = basket_record.get("eigenvector", [])

        # Build derived symbol representations for the different subsystems.
        symbols = [str(sym).upper() for sym in basket_symbols]
        symbols_usdt = [f"{sym}-USDT" for sym in symbols]
        symbols_perp = [f"{sym}-PERP" for sym in symbols]

        config = {
            "name": f"basket_{idx + 1}_{'_'.join(symbols)}",
            "symbols": symbols,
            "symbols_usdt": symbols_usdt,
            "symbols_perp": symbols_perp,
            "eigenvector": list(eigenvector),
            "entry_threshold": float(optimal_params.get("entry_threshold", 1.5)),
            "exit_threshold": float(optimal_params.get("exit_threshold", 0.5)),
            "lookback_days": int(optimal_params.get("lookback_days", 30)),
            "max_alloc_frac": float(
                basket_record.get("max_alloc_frac", default_max_alloc_frac)
            ),
            "timeframe": timeframe,
            "price_type": price_type,
            "metadata": {
                "half_life_days": float(basket_record.get("half_life_days", 0.0)),
                "johansen_p_value": float(
                    basket_record.get("johansen_p_value", float("nan"))
                ),
                "hurst_half_life_days": float(basket_record.get("hurst_half_life_days", float("inf"))),
                "sustainability_rolling": float(
                    basket_record.get("sustainability_rolling", float("nan"))
                ),
                "sustainability_discrete": float(
                    basket_record.get("sustainability_discrete", float("nan"))
                ),
                "optimization_stats": {
                    key: value
                    for key, value in optimal_params.items()
                    if key not in {"entry_threshold", "exit_threshold", "lookback_days"}
                },
            },
        }

        combined_configs.append(config)

    return combined_configs


__all__ = ["load_cointegration_basket_configs"]


