"""
Visualization and reporting for cointegration results.
"""

from typing import Dict
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


def plot_spread_analysis(best_basket: Dict, price_data: pd.DataFrame) -> None:
    """
    Plot spread series and z-scores.
    
    Parameters:
    -----------
    best_basket : Dict
        Best basket result dictionary
    price_data : pd.DataFrame
        Price data with timestamp index
    """
    spread = best_basket['spread']
    timestamps = price_data.index[:len(spread)]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Spread Series', 'Spread Z-Score (Rolling)'),
        vertical_spacing=0.1
    )
    
    # Spread series
    fig.add_trace(
        go.Scatter(x=timestamps, y=spread, name='Spread', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Z-score
    lookback_bars = best_basket['optimal_lookback'] * 96
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
    
    fig.add_trace(
        go.Scatter(x=z_timestamps, y=z_scores, name='Z-Score', line=dict(color='red')),
        row=2, col=1
    )
    # Use size-dependent threshold if available
    basket_size = len(best_basket.get('basket', []))
    z_threshold = best_basket.get('z_threshold', 3.0 + 0.5 * max(0, basket_size - 2))
    fig.add_hline(y=z_threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"{z_threshold:.1f}σ limit", row=2, col=1)
    fig.add_hline(y=-z_threshold, line_dash="dash", line_color="red", row=2, col=1)
    
    fig.update_layout(
        title=f"Best Basket: {', '.join(best_basket['basket'])}",
        height=600,
        showlegend=True
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Spread", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    
    fig.show()


def plot_lookback_optimization(best_basket: Dict) -> None:
    """
    Plot lookback window optimization results.
    
    Parameters:
    -----------
    best_basket : Dict
        Best basket result dictionary with 'lookback_results'
    """
    if 'lookback_results' not in best_basket:
        return
    
    results = best_basket['lookback_results']
    windows = [r['window_days'] for r in results]
    persistences = [r['persistence'] for r in results]
    max_zs = [r['max_zscore'] for r in results]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Persistence vs Lookback', 'Max Z-Score vs Lookback')
    )
    
    fig.add_trace(
        go.Scatter(x=windows, y=persistences, mode='lines+markers', name='Persistence'),
        row=1, col=1
    )
    # Use size-dependent threshold if available, otherwise default
    basket_size = len(best_basket.get('basket', []))
    persistence_threshold = best_basket.get('persistence_threshold', 0.8 - 0.1 * max(0, basket_size - 2))
    z_threshold = best_basket.get('z_threshold', 3.0 + 0.5 * max(0, basket_size - 2))
    
    fig.add_hline(y=persistence_threshold, line_dash="dash", line_color="green", 
                  annotation_text=f"{persistence_threshold:.1f} threshold", row=1, col=1)
    
    fig.add_trace(
        go.Scatter(x=windows, y=max_zs, mode='lines+markers', name='Max Z-Score', line=dict(color='red')),
        row=1, col=2
    )
    fig.add_hline(y=z_threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"{z_threshold:.1f}σ limit", row=1, col=2)
    
    fig.update_layout(
        title=f"Lookback Window Optimization: {', '.join(best_basket['basket'])}",
        height=400
    )
    fig.update_xaxes(title_text="Lookback (days)", row=1, col=1)
    fig.update_xaxes(title_text="Lookback (days)", row=1, col=2)
    fig.update_yaxes(title_text="Persistence Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Max Z-Score", row=1, col=2)
    
    fig.show()


def print_summary_statistics(best_basket: Dict) -> None:
    """
    Print summary statistics for the best basket.
    
    Parameters:
    -----------
    best_basket : Dict
        Best basket result dictionary
    """
    spread = best_basket['spread']
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Best Basket: {', '.join(best_basket['basket'])}")
    print(f"Basket Size: {len(best_basket['basket'])}")
    print(f"\nCointegration Test:")
    print(f"  Trace Statistic: {best_basket['johansen_result']['trace_stat']:.4f}")
    print(f"  Critical Value (5%): {best_basket['johansen_result']['trace_crit']:.4f}")
    print(f"  P-value: {best_basket['johansen_result']['p_value']:.4f}")
    print(f"\nPersistence Metrics:")
    print(f"  Persistence Ratio: {best_basket['persistence']:.3f}")
    if 'persistence_threshold' in best_basket:
        print(f"  Threshold Used: {best_basket['persistence_threshold']:.2f}")
    print(f"\nSpread Statistics:")
    print(f"  Mean: {np.mean(spread):.6f}")
    print(f"  Std: {np.std(spread):.6f}")
    print(f"  Max Z-Score: {best_basket['max_zscore']:.2f}")
    if 'z_threshold' in best_basket:
        print(f"  Z-Score Threshold Used: {best_basket['z_threshold']:.2f}")
    print(f"\nOptimal Parameters:")
    print(f"  Lookback Window: {best_basket['optimal_lookback']} days")
    print(f"  Final Score: {best_basket['final_score']:.3f}")
    print(f"\nCointegration Vector (normalized):")
    eigenvector = best_basket['eigenvector']
    for i, sym in enumerate(best_basket['basket']):
        print(f"  {sym}: {eigenvector[i]:.6f}")
    print("="*60)

