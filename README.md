# HKU-Avenir Quantitative Trading Challenge -- Round 2

Live trading phase of the HKU x Avenir Web3.0 Quantitative Trading Competition. Round 2 ran for 12 weeks on Binance perpetual futures. Our team advanced from Round 1.

## What this repo contains

- **Backtester engine** -- event-driven loop that replays 15-minute OHLCV bars through a strategy and position manager, tracks PnL via a simulated OMS, and supports permutation testing for statistical significance (p-value from shuffled price series).
- **Hypothesis testing pipeline** -- Johansen cointegration tests on multi-asset baskets, followed by sustainability filtering (rolling/discrete persistence), Hurst-exponent half-life filtering, and z-score parameter optimization (grid search over entry/exit thresholds and lookback windows, ranked by Sharpe).
- **HMM regime detection** -- two-state Hidden Markov Model (hmmlearn) fitted per symbol on log returns, used to mask cointegration tests to low-volatility regimes.
- **Three trading strategies** with paired position managers:
  - `v1_ls` -- Long BTC / short altcoins with negative 24h returns, variance-adjusted weights, daily rebalance.
  - `v1_pairs` / `v2_pairs` -- Classical pairs trading (OLS beta on log prices, z-score entry/exit).
  - `v3_cointegration` -- Basket cointegration using Johansen eigenvector weights and optimized z-score thresholds.
- **Position managers** -- risk controls including volatility screening, inverse-vol sizing, 15% stop-loss, allocation ramp-up, portfolio optimization via PyPortfolioOpt (min-variance / max-Sharpe), and threshold-based rebalancing.

## Project structure

```
backtester/
  src/
    backtester.py          # Main backtest loop + permutation testing
    oms_simulation.py      # Simulated OMS (positions, PnL, trade history)
    hist_data.py           # Historical data collection from Binance via CCXT
    format_utils.py        # Logging table formatters
    generate_regime_labels.py
    utils.py
  backtest/
    strategies/
      v1_ls.py             # Long BTC / short altcoin strategy
      v1_pairs.py          # Pairs trading v1
      v2_pairs.py          # Pairs trading v2 (with logging)
      v3_cointegration.py  # Basket cointegration strategy
      cointegration_loader.py
    position_managers/
      example.py           # Inverse-vol sizing PM
      v1_ls_pm.py          # LS strategy PM (ramp-up, stop-loss)
      v2_pairs_pm.py       # Pairs PM (PyPortfolioOpt)
      v3_pairs_pm.py       # Pairs PM with rebalancing
    v1_ls_bt.py            # Runner scripts
    v2_pairs_bt.py
    v3_cointegration_bt.py
    example/
      example.py
      v1_hold.py
      test_multiprocess_backtest.py

hypothesis_testing/
  cointegration/
    johansen_test.py       # Johansen trace test wrapper
    hmm_regimes.py         # HMM regime labeling
    basket_generator.py    # Clustering + combinatorial basket generation
    data_loader.py         # Parallel parquet loader
    data_split.py          # Chronological train/test splitting
    filter_sustainability.py
    filter_mean_reversion.py
    deduplicate_baskets.py
    optimize_zscore.py     # Z-score parameter grid search
    plot_spreads.py
    visualization.py
    utils_parallel.py
    test_cointegration_framework.ipynb
    hmm_regime_hook.ipynb
```

## Tech stack

Python 3.12, ccxt, pandas, numpy, statsmodels (Johansen test), hmmlearn, scikit-learn, scipy, PyPortfolioOpt, plotly, matplotlib.
