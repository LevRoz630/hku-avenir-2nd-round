"""
Standalone script to plot spreads on test data from validated baskets.
Reads baskets from validated_baskets.json and test data from parquet files.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Find workspace root
current = Path(__file__).resolve().parent
while current.name != 'hku-datawork' and current.parent != current:
    current = current.parent
workspace_root = current if current.name == 'hku-datawork' else Path('/workspace-hku/hku-datawork')
sys.path.insert(0, str(workspace_root))

from hypothesis_testing.cointegration.data_loader import load_price_data
from hypothesis_testing.cointegration.data_split import split_data_chronologically


def plot_spread_on_test_data(basket_data, test_data, basket_idx):
    basket = basket_data['basket']
    eigenvector = np.array(basket_data['eigenvector'])
    
    # Extract basket prices from test_data
    basket_cols = [f'{sym}_close' for sym in basket]
    missing_cols = [col for col in basket_cols if col not in test_data.columns]
    if missing_cols:
        print(f"Warning: Missing columns for basket {basket}: {missing_cols}")
        return
    
    basket_prices = test_data[basket_cols].values
    
    # Convert to log prices
    log_prices = np.log(basket_prices)
    
    # Compute spread using eigenvector
    spread = log_prices @ eigenvector
    
    # Get timestamps
    timestamps = test_data.index[:len(spread)]
    
    # Get stats from JSON
    half_life_days = basket_data.get('half_life_days', None)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot spread series
    ax1.plot(timestamps, spread, 'b-', linewidth=0.8, label='Spread')
    spread_mean = np.mean(spread)
    ax1.axhline(y=spread_mean, color='gray', linestyle='--', linewidth=1.5, 
                label=f'Mean: {spread_mean:.4f}')
    ax1.set_ylabel('Spread', fontsize=12)
    ax1.set_title(f"Spread Series (Test Data) - Basket {basket_idx+1}: {', '.join(basket)}", 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Compute rolling z-score (30 days = 30 * 96 bars for 15m data)
    bars_per_day = 96  # 15m timeframe
    lookback_bars = 30 * bars_per_day
    z_scores = []
    z_timestamps = []
    for i in range(lookback_bars, len(spread)):
        window_spread = spread[i - lookback_bars:i]
        mean = np.mean(window_spread)
        std = np.std(window_spread)
        if std > 0:
            z = (spread[i] - mean) / std
            z_scores.append(z)
            z_timestamps.append(timestamps[i])
    
    # Plot z-score
    if z_scores:
        ax2.plot(z_timestamps, z_scores, 'r-', linewidth=0.8, label='Z-Score')
        ax2.axhline(y=2.0, color='orange', linestyle='--', linewidth=1.5, label='±2σ')
        ax2.axhline(y=-2.0, color='orange', linestyle='--', linewidth=1.5)
        ax2.axhline(y=3.0, color='red', linestyle='--', linewidth=1.5, label='±3σ')
        ax2.axhline(y=-3.0, color='red', linestyle='--', linewidth=1.5)
        ax2.fill_between(z_timestamps, -2, 2, alpha=0.1, color='green', label='±2σ zone')
    
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Z-Score', fontsize=12)
    ax2.set_title('Spread Z-Score (Rolling 30 days)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # Add stats to title
    stats_text = ""
    if half_life_days is not None:
        if np.isinf(half_life_days):
            stats_text = "Half-life: inf"
        else:
            stats_text = f"Half-life: {half_life_days:.1f} days"
    stats_text += " | Hurst-based half-life"
    
    if stats_text:
        fig.suptitle(stats_text, fontsize=11, y=0.995)
    plt.tight_layout()
    
    # Save plot to file
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    basket_name = "_".join(basket)
    output_file = output_dir / f"spread_basket_{basket_idx+1}_{basket_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_file}")
    plt.close()


def main():
    """Main function to load data and plot spreads."""
    # Configuration - match notebook settings
    DATA_DIR = Path("/workspace-hku/hku-data/test_data")
    TIMEFRAME = '15m'
    PRICE_TYPE = 'mark'
    VALIDATED_BASKETS_FILE = Path(__file__).parent / "validated_baskets.json"
    
    # Load validated baskets
    print(f"Loading validated baskets from {VALIDATED_BASKETS_FILE}...")
    with open(VALIDATED_BASKETS_FILE, 'r') as f:
        data = json.load(f)
    
    baskets = data['baskets']
    config = data['config']
    print(f"Found {len(baskets)} validated baskets")
    print(f"Config: timeframe={config['timeframe']}, price_type={config['price_type']}\n")
    
    # Load price data
    print(f"Loading price data from {DATA_DIR}...")
    price_data = load_price_data(
        data_dir=DATA_DIR,
        years_back=1.2,
        symbols=None,
        timeframe=TIMEFRAME,
        price_type=PRICE_TYPE,
        max_workers=None
    )
    print(f"Loaded {len(price_data)} timestamps for {len(price_data.columns)} symbols\n")
    
    # Split data to get test_data (matching notebook splits)
    print("Splitting data chronologically...")
    splits = split_data_chronologically(
        price_data,
        cluster_split=(1.2, 0.8),
        cointegration_split=(0.8, 0.2),
        test_split=(0.2, 0.0)
    )
    test_data = splits['test_data']
    print(f"Test data: {len(test_data)} samples from {test_data.index.min()} to {test_data.index.max()}\n")
    
    # Plot spreads for all baskets
    print(f"Plotting spreads for {len(baskets)} baskets on test data...\n")
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    print(f"Plots will be saved to: {output_dir}\n")
    
    for i, basket_data in enumerate(baskets):
        print(f"Basket {i+1}/{len(baskets)}: {', '.join(basket_data['basket'])}")
        plot_spread_on_test_data(basket_data, test_data, i)
    
    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    main()

