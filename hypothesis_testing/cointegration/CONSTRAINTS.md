# Cointegration Testing Constraints and Regions

## Constraints Applied in Parallel Testing

### Step 1: Initial Cointegration Test (`test_baskets_cointegration_parallel`)

**Constraint**: Johansen Trace Test
- **Threshold**: `p-value < 0.05` (hard constraint)
- **Test**: Tests H0: rank <= 0 (no cointegration) vs H1: rank > 0 (cointegration exists)
- **Method**: Uses `coint_johansen` from statsmodels with:
  - `det_order = -1` (no deterministic terms)
  - `k_ar_diff = 1` (1 lag in VAR model)
- **Minimum Data Requirement**: 
  - Need at least `10 * n` observations where `n` = number of assets in basket
  - Example: For basket of 2 assets, need at least 20 observations
  - Example: For basket of 4 assets, need at least 40 observations

**What Gets Filtered Out**:
- Baskets where Johansen test p-value >= 0.05
- Baskets with insufficient data (< 10 observations per asset)
- Baskets where trace statistic < critical value (5% level)

### Step 2: Sustainability Filter (`filter_baskets_sustainability`)

**Constraints**:
- **Rolling Windows**: 
  - Window size: `WINDOW_DAYS` (default: 90 days)
  - Step size: `STEP_DAYS` (default: 30 days)
  - Persistence threshold: `PERSISTENCE_THRESHOLD` (default: 0.7 = 70%)
  - **Requirement**: At least 70% of rolling windows must pass cointegration test
  
- **Discrete Periods**:
  - Period size: `PERIOD_DAYS` (default: 90 days)
  - Persistence threshold: `PERSISTENCE_THRESHOLD` (default: 0.7 = 70%)
  - **Requirement**: At least 70% of discrete periods must pass cointegration test

**What Gets Filtered Out**:
- Baskets where < 70% of rolling windows pass cointegration
- Baskets where < 70% of discrete periods pass cointegration
- If both rolling and discrete are enabled, BOTH must pass threshold

### Step 3: Mean Reversion Filter (`filter_baskets_mean_reversion`)

**Constraints**:
- **Half-life Threshold**: `HALF_LIFE_THRESHOLD_DAYS` (default: 30.0 days)
  - **Requirement**: Half-life < 30 days OR ADF p-value < 0.01
- **ADF Test**: 
  - Stationarity threshold: `p-value < 0.01` (strong stationarity signal)
  - Uses automatic lag selection (AIC)

**What Gets Filtered Out**:
- Baskets with half-life >= 30 days AND ADF p-value >= 0.01
- Non-stationary spreads (theta <= 0 or theta >= 1 in AR(1) model)

## Testing Regions

### Data Coverage
- **Full Dataset**: All available timestamps in loaded price data
- **Time Range**: From `price_data.index.min()` to `price_data.index.max()`
- **Example**: If data spans 2025-03-16 to 2025-10-11, entire period is tested

### Rolling Window Testing
- **Windows**: Overlapping windows across full dataset
- **Window Size**: 90 days × `BARS_PER_DAY` bars
  - For 15m timeframe: 90 days × 96 bars/day = 8,640 bars per window
  - For 1h timeframe: 90 days × 24 bars/day = 2,160 bars per window
- **Step Size**: 30 days × `BARS_PER_DAY` bars
  - For 15m: 30 days × 96 = 2,880 bars step
  - For 1h: 30 days × 24 = 720 bars step
- **Coverage**: Tests every 30-day interval across full dataset

### Discrete Period Testing
- **Periods**: Non-overlapping periods across full dataset
- **Period Size**: 90 days × `BARS_PER_DAY` bars
- **Coverage**: Tests consecutive 90-day periods from start to end

## Why You Might Get 0 Valid Baskets

### Common Failure Reasons:

1. **Too Strict Johansen Test**:
   - p-value threshold of 0.05 may be too strict for crypto markets
   - Try relaxing to 0.10 or checking trace statistic vs critical value

2. **Insufficient Data**:
   - Need at least 10 observations per asset
   - Check: `len(price_data) >= 10 * max_basket_size`

3. **Data Quality Issues**:
   - Missing data after alignment (>5% missing gets filtered)
   - Check: `price_data.isna().sum()` before testing

4. **Market Regime Changes**:
   - Crypto markets may not have stable cointegration relationships
   - Try testing on shorter time periods or different market regimes

5. **Basket Generation Issues**:
   - Clustering may not generate good candidate baskets
   - Check: How many candidate baskets are generated?

## Debugging Steps

1. **Check Data Load**:
   ```python
   print(f"Data shape: {price_data.shape}")
   print(f"Date range: {price_data.index.min()} to {price_data.index.max()}")
   print(f"Missing data: {price_data.isna().sum().sum()}")
   ```

2. **Check Candidate Baskets**:
   ```python
   print(f"Number of candidate baskets: {len(candidate_baskets)}")
   print(f"Sample baskets: {candidate_baskets[:5]}")
   ```

3. **Check Johansen Test Results**:
   - Add logging to see p-values before filtering
   - Check if any baskets have p-value close to 0.05

4. **Relax Constraints Temporarily**:
   - Try p-value < 0.10 instead of 0.05
   - Check how many pass with relaxed threshold

5. **Test on Subset**:
   - Test on smaller time period first
   - Test on fewer symbols to isolate issues

## Current Configuration (from notebook)

- **Timeframe**: 15m
- **Price Type**: mark
- **Bars per day**: 96
- **Persistence threshold**: 0.7 (70%)
- **Window days**: 90
- **Step days**: 30
- **Period days**: 90
- **Half-life threshold**: 30.0 days
- **Johansen p-value threshold**: 0.05 (hardcoded in `johansen_test.py`)

