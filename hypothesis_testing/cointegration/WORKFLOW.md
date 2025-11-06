# Cointegration Testing Workflow

## Overview
This workflow identifies cryptocurrency baskets suitable for pairs trading strategies with stable long-term cointegration relationships. **Critical**: Data is split chronologically to prevent overfitting and data leakage.

## Data Splitting Strategy

To prevent overfitting, data is split chronologically into three separate datasets:

1. **Cluster Data** (1.0-0.8 quantile): Most recent 20% of data
   - Used for: Basket generation via clustering
   - Why: Uses most recent market structure to find correlated assets

2. **Cointegration Data** (0.8-0.2 quantile): Middle 60% of data  
   - Used for: Initial cointegration testing and sustainability filtering
   - Why: Tests baskets on unseen data (not used for clustering)

3. **Half-life Data** (0.2-0.0 quantile): Oldest 20% of data
   - Used for: Mean reversion speed testing
   - Why: Final validation on completely separate data

## Pipeline Steps

### Step 1: Load Price Data
- Load historical price data from parquet files
- Align timestamps across all symbols
- Filter symbols with >5% missing data

### Step 2: Split Data Chronologically
- Split into cluster_data, cointegration_data, half_life_data
- Ensures no data leakage between steps

### Step 3: Generate Candidate Baskets
- **Input**: cluster_data (1.2 - 0.8)
- **Process**: Hierarchical clustering on correlation matrix
- **Output**: List of candidate basket combinations
- **Why separate data**: Prevents overfitting - baskets generated on different data than tested

### Step 4: Test Initial Cointegration
- **Input**: candidate_baskets, cointegration_data (middle 60%)
- **Process**: Johansen trace test (p-value < 0.01)
- **Output**: Cointegrated baskets with eigenvectors
- **Deduplication**: Remove baskets with >50% overlap

### Step 5: Filter by Sustainability
- **Input**: cointegrated_baskets, cointegration_data
- **Process**: 
  - Rolling windows: Test cointegration across overlapping windows
  - Discrete periods: Test cointegration across non-overlapping periods
- **Filter**: ≥70% of windows/periods must pass cointegration test
- **Output**: Sustainable baskets

### Step 6: Filter by Mean Reversion Speed
- **Input**: sustainable_baskets, half_life_data (oldest 20%)
- **Process**: 
  - Recompute spreads from half_life_data using eigenvectors
  - Compute half-life and ADF test
- **Filter**: Half-life < 30 days OR ADF p-value < 0.01
- **Output**: Fast mean-reverting baskets
- **Why separate data**: Prevents data leakage - tests on completely unseen data

### Step 7: Save Validated Baskets
- Export to JSON for strategy deployment

## Key Improvements

1. **No Overfitting**: Each step uses different data
2. **No Look-ahead Bias**: Clustering uses recent data, testing uses older data
3. **No Data Leakage**: Half-life computed on separate dataset
4. **Realistic Validation**: Tests baskets on truly unseen data

## Configuration

- **P-value threshold**: 0.01 (1% significance)
- **Persistence threshold**: 0.7 (70% of windows must pass)
- **Half-life threshold**: 30 days
- **Max combinations per cluster**: 4000
- **Workers**: 90% of CPU count (leaves headroom for OS)
