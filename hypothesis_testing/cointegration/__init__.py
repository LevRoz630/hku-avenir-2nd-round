"""
Basket cointegration hypothesis testing modules.
"""

from .data_loader import load_price_data
from .johansen_test import johansen_test
from .basket_generator import generate_baskets_clustering, test_basket_cointegration, compute_spread
from .persistence import test_persistence_rolling
from .spread_validator import compute_max_zscore, validate_spread_constraints
from .optimizer import optimize_lookback_window, score_basket, optimize_basket_size
from .visualization import plot_spread_analysis, plot_lookback_optimization, print_summary_statistics

__all__ = [
    'load_price_data',
    'johansen_test',
    'generate_baskets_clustering',
    'test_basket_cointegration',
    'compute_spread',
    'test_persistence_rolling',
    'compute_max_zscore',
    'validate_spread_constraints',
    'optimize_lookback_window',
    'score_basket',
    'optimize_basket_size',
    'plot_spread_analysis',
    'plot_lookback_optimization',
    'print_summary_statistics',
]

