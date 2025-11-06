#!/usr/bin/env python3
"""Run the v3 cointegration basket strategy in the backtester."""

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


# Ensure local strategy modules and engine src are importable
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR.parent / "src"))

from backtester import Backtester  # type: ignore  # noqa: E402
from position_managers.v1_ls_pm import V1LSPositionManager  # type: ignore  # noqa: E402
from strategies.cointegration_loader import (  # type: ignore  # noqa: E402
    load_cointegration_basket_configs,
)
from strategies.v3_cointegration import V3CointegrationStrategy  # type: ignore  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _resolve_default_paths() -> tuple[Path, Path, Path]:
    """Return paths for JSON configs and historical data directory."""
    repo_root = SCRIPT_DIR.parent.parent
    hypothesis_dir = repo_root / "hypothesis_testing" / "cointegration"
    validated_path = hypothesis_dir / "validated_baskets.json"
    optimization_path = hypothesis_dir / "zscore_optimization.json"
    hist_dir = repo_root / "hku-data" / "test_data"
    return validated_path, optimization_path, hist_dir


def main() -> None:
    validated_path, optimization_path, hist_dir = _resolve_default_paths()

    logger.info("Loading basket configs from %s and %s", validated_path, optimization_path)
    basket_configs = load_cointegration_basket_configs(
        validated_path=str(validated_path),
        optimization_path=str(optimization_path),
    )

    if not basket_configs:
        raise RuntimeError("No basket configs available; ensure hypothesis tests have been run.")

    logger.info("Loaded %d baskets", len(basket_configs))

    strategy = V3CointegrationStrategy(
        baskets=basket_configs,
        historical_data_dir=str(hist_dir),
        timeframe="15m",
    )

    position_manager = V1LSPositionManager()
    backtester = Backtester(historical_data_dir=str(hist_dir))

    # Run over the last ~90 days by default
    end_date = datetime.now(timezone.utc) - timedelta(days=3)
    start_date = end_date - timedelta(days=30)

    logger.info("Running backtest from %s to %s", start_date, end_date)

    results = backtester.run_backtest(
        strategy=strategy,
        position_manager=position_manager,
        start_date=start_date,
        end_date=end_date,
        time_step=timedelta(minutes=15),
        market_type="futures",
    )

    backtester.print_results(results)


if __name__ == "__main__":
    main()


