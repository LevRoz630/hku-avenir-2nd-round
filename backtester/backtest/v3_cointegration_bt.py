#!/usr/bin/env python3
"""Run the v3 cointegration basket strategy in the backtester."""

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


# Ensure local strategy modules and engine src are importable
SCRIPT_DIR = Path(__file__).resolve().parent
BACKTEST_ROOT = SCRIPT_DIR
REPO_ROOT = SCRIPT_DIR.parents[2]

sys.path.append(str(BACKTEST_ROOT))
sys.path.append(str(REPO_ROOT / "backtester" / "src"))

from backtester import Backtester  # type: ignore  # noqa: E402
from position_managers.v3_pairs_pm import PositionManager  # type: ignore  # noqa: E402
from strategies.cointegration_loader import (  # type: ignore  # noqa: E402
    load_cointegration_basket_configs,
)
from strategies.v3_cointegration import V3CointegrationStrategy  # type: ignore  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _resolve_default_paths() -> tuple[Path, Path, Path]:
    """Return paths for JSON configs and historical data directory."""

    hypothesis_dir = REPO_ROOT / "hypothesis_testing" / "cointegration"
    validated_path = hypothesis_dir / "validated_baskets.json"
    optimization_path = hypothesis_dir / "zscore_optimization.json"
    hist_dir = REPO_ROOT / "hku-data" / "test_data"
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

    # Get max lookback_days from strategy (baskets have lookback_days of 30-45)
    max_basket_lookback = strategy.lookback_days if strategy else 30
    min_lookback_days = max(max_basket_lookback, 30)

    position_manager = PositionManager(
        portfolio_alloc_frac=0.8,
        risk_method='min_volatility',
        max_total_allocation=2000.0,
        min_lookback_days=min_lookback_days,
    )
    backtester = Backtester(historical_data_dir=str(hist_dir))

    # Run over the last ~90 days by default
    end_date = datetime.now(timezone.utc) - timedelta(days=1)
    start_date = end_date - timedelta(days=90)

    logger.info("Running backtest from %s to %s", start_date, end_date)

    results = backtester.run_backtest(
        strategy=strategy,
        position_manager=position_manager,
        start_date=start_date,
        end_date=end_date,
        time_step=timedelta(days=1),
        market_type="futures",
    )

    backtester.print_results(results)


if __name__ == "__main__":
    main()


