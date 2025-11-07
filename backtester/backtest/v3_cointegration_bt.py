#!/usr/bin/env python3
"""Run the v3 cointegration basket strategy in the backtester."""

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


# Ensure local strategy modules and engine src are importable
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))  # For 'from src.*' imports
sys.path.append(str(Path(__file__).parent.parent / "src"))  # For 'from backtester' import

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
    repo_root = Path(__file__).parent.parent.parent
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

    logger.info(f"Strategy symbols: {strategy.symbols}")

    # Get lookback days from strategy (baskets need at least 30 days)
    lookback_days = max(strategy.lookback_days, 30)

    position_manager = PositionManager(
        risk_method='max_sharpe',
        max_total_allocation=500.0,
        min_lookback_days=lookback_days,
    )
    backtester = Backtester(historical_data_dir=str(hist_dir))

    # Run over the last 2 days by default
    end_date = datetime.now(timezone.utc) - timedelta(days=90)
    start_date = end_date - timedelta(days=3)

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


    backtester.save_results(results, "v3_cointegration_bt")

    # Generate and plot regime data for comparison (if sufficient data quality)
    # This shows HMM volatility regimes (green=low vol, red=high vol) alongside performance
    backtester.load_regime_data()  # Always try to generate, may be empty for short periods
    backtester.plot_regimes()  # Silently skips if no valid regime data

    backtester.plot_portfolio_value()
    backtester.plot_drawdown()
    backtester.plot_returns()
    backtester.plot_positions()


if __name__ == "__main__":
    main()
