# Cointegration Testing Workflow

## Purpose
Identify cryptocurrency baskets suitable for pairs trading strategies with stable long-term cointegration relationships. The workflow filters candidate baskets through multiple statistical tests to ensure they exhibit persistent cointegration and fast mean reversion.

## Inputs

### Data Files
- **Location**: `hku-data/test_data/`
- **Format**: Parquet files ONLY (CSV files are ignored)
- **Pattern**: `perpetual_{SYMBOL}_{index|mark}_{timeframe}_*.parquet`
  - Example: `perpetual_BTC_index_1h_20240101_20241231.parquet`
  - Example: `perpetual_ETH_mark_15m_20240101_20241231.parquet`
- **Columns**: Must contain `timestamp` and `close` columns
- **Data Type**: Perpetual futures price data (index or mark prices)

### Configuration Parameters
- `timeframe`: '1h' (index) or '15m' (mark)
- `price_type`: 'index' or 'mark'
- `symbols`: Optional list to filter, or None to auto-discover all
- `MIN_BASKET_SIZE`: Minimum number of assets in basket (default: 2)
- `MAX_BASKET_SIZE`: Maximum number of assets in basket (default: 4)
- `N_CLUSTERS`: Number of clusters for basket generation (default: 5)
- `PERSISTENCE_THRESHOLD`: Minimum persistence ratio (default: 0.7 = 70%)
- `HALF_LIFE_THRESHOLD_DAYS`: Maximum half-life in days (default: 30.0)

## Pipeline Steps

### 1. Load Price Data
- **Input**: Parquet files from `hku-data/test_data/`
- **Process**: 
  - Parallel loading of matching parquet files
  - Timestamp alignment across symbols
  - Forward fill gaps (max 1 day)
  - Filter symbols with >5% missing data
- **Output**: Aligned DataFrame with columns `{symbol}_close`, timestamp index

### 2. Generate Candidate Baskets
- **Input**: Price data DataFrame
- **Process**: Clustering-based basket generation (parallel)
- **Output**: List of candidate basket symbol combinations

### 3. Test Initial Cointegration
- **Input**: Candidate baskets, price data
- **Process**: Johansen trace test (multiprocessed, batch processing)
- **Filter**: Only baskets passing cointegration test (p-value threshold)
- **Output**: List of cointegrated baskets with eigenvectors and test statistics

### 4. Filter by Sustainability
- **Input**: Cointegrated baskets, price data
- **Process**: 
  - Rolling window test: Test cointegration across overlapping windows
  - Discrete period test: Test cointegration across non-overlapping periods
- **Filters**:
  - `PERSISTENCE_THRESHOLD`: Minimum ratio of windows/periods that must pass (default: 70%)
  - `WINDOW_DAYS`: Rolling window size (default: 90 days)
  - `STEP_DAYS`: Rolling window step size (default: 30 days)
  - `PERIOD_DAYS`: Discrete period size (default: 90 days)
- **Output**: Sustainable baskets with persistence ratios

### 5. Filter by Mean Reversion Speed
- **Input**: Sustainable baskets, price data
- **Process**: 
  - Compute spread time series
  - Estimate half-life using AR(1) model
  - ADF test for stationarity
- **Filter**: `HALF_LIFE_THRESHOLD_DAYS`: Maximum half-life (default: 30 days)
- **Output**: Fast mean-reverting baskets with half-life statistics

### 6. Output Validated Baskets
- **Input**: Fast mean-reverting baskets
- **Process**: Serialize to JSON format
- **Output**: `validated_baskets.json` file

## Outputs

### validated_baskets.json
JSON file containing:
- `timestamp`: Generation timestamp
- `config`: Configuration parameters used
- `baskets`: Array of validated baskets, each with:
  - `basket`: List of symbol names
  - `eigenvector`: Cointegration eigenvector (weights)
  - `johansen_p_value`: Johansen test p-value
  - `johansen_trace_stat`: Johansen trace statistic
  - `sustainability_rolling`: Rolling windows persistence ratio
  - `sustainability_discrete`: Discrete periods persistence ratio
  - `half_life_days`: Mean reversion half-life in days
  - `adf_p_value`: Augmented Dickey-Fuller test p-value
  - `is_stationary`: Stationarity flag

## Filters Summary

1. **Cointegration Filter**: Johansen trace test - ensures long-term equilibrium relationship exists
2. **Sustainability Filter**: Persistence across time windows - ensures relationship is stable, not just in-sample
3. **Mean Reversion Filter**: Half-life threshold - ensures spread reverts quickly enough for profitable trading

## Usage Example

See `test_cointegration_framework.ipynb` for complete implementation example.

