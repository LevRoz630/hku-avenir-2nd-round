from typing import List, Dict, Any, Optional
import logging
from datetime import timedelta
from src.hist_data import HistoricalDataCollector
from src.oms_simulation import OMSClient
import pandas as pd
from pypfopt import risk_models, expected_returns, EfficientFrontier

logger = logging.getLogger(__name__)
class PositionManager:
    """
    Pair-level position manager for pairs trading.
    
    Manages risk at the pair level, not individual symbols:
    1. Groups orders by pair_id
    2. Uses pyportfolioopt for portfolio-level risk management 
    3. Allocates capital to pairs based on risk parity
    4. Sizes each pair's legs using beta to ensure net beta neutrality
    """
    
    def __init__(self, 
                 portfolio_alloc_frac: float = 0.8,
                 risk_method: str = 'min_volatility',
                 min_lookback_days: int = 30):
        """
        Args:
            portfolio_alloc_frac: Maximum fraction of balance to allocate to all pairs (default 0.8 = 80%)
            risk_method: 'min_volatility' or 'max_sharpe'
            min_lookback_days: Minimum days of spread returns needed for risk calculation
        """
        self.oms_client = None
        self.data_manager = None
        self.portfolio_alloc_frac = portfolio_alloc_frac
        self.risk_method = risk_method
        self.min_lookback_days = min_lookback_days
        self.pair_spread_history = {}  # Track spread returns for each pair
        
    def filter_orders(self, orders: List[Dict[str, Any]], 
                     oms_client: OMSClient, 
                     data_manager: HistoricalDataCollector) -> Optional[List[Dict[str, Any]]]:
        """
        Main entry point for filtering and sizing orders.
        
        Steps:
        1. Separate CLOSE orders (pass through immediately)
        2. Group OPEN orders by pair_id
        3. Compute spread returns for each pair
        4. Use pyportfolioopt to allocate capital across pairs (risk parity/min vol)
        5. Size each pair's legs using beta to ensure net beta neutrality
        6. Return sized orders
        """
        self.oms_client = oms_client
        self.data_manager = data_manager
        self.pairs_dict = self._group_orders_by_pair(orders)
        
        try:
            # Same filtering of close orders and open orders as in example
            close_orders = [o for o in orders if o.get('side') == 'CLOSE']
            open_orders = [o for o in orders if o.get('side') != 'CLOSE']
            
            if not open_orders:
                return close_orders if close_orders else None
            
            # 1. Allocate capital to pairs based on risk management strat
            pair_allocations = self._size_pairs(self.pairs_dict, oms_client.balance['USDT'])
            
            if not pair_allocations:
                logger.warning("No capital allocated to any pairs")
                return close_orders if close_orders else None
            
            # 2. Using the allocated capital per each pair, size the orders for each leg of the pair based on beta to ensure market neutral net beta
            sized_orders = self._size_orders(self.pairs_dict, pair_allocations)
            
            # Bam
            return close_orders + sized_orders
            
        except Exception as e:
            logger.error(f"Error filtering orders: {e}", exc_info=True)
            return None
    
    def _group_orders_by_pair(self, orders: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group orders by pair_id."""
        pairs_dict = {}
        for order in orders:
            pair_id = order.get('pair_id')
            if not pair_id:
                logger.warning(f"Order missing pair_id: {order}")
                continue
            
            if pair_id not in pairs_dict:
                pairs_dict[pair_id] = []
            pairs_dict[pair_id].append(order)
        
        return pairs_dict
    
    def _compute_spread_returns(self, pairs_dict: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Compute historical spread returns for each pair.
        Returns DataFrame with columns = pair_ids, rows = dates
        """
        spread_returns_dict = {}
        
        for pair_id, orders in pairs_dict.items():
            # Extract beta from ratio field and identify symbols
            # Filter out CLOSE orders
            open_orders = [o for o in orders if o.get('side') != 'CLOSE' and o.get('ratio') is not None]
            
            if len(open_orders) < 2:
                logger.warning(f"Pair {pair_id} has insufficient open orders")
                continue
            
            # Extract beta from ratio field
            # For short spread: A has ratio=1, B has ratio=beta → beta = max(ratios)
            # For long spread: A has ratio=beta, B has ratio=1 → beta = max(ratios)
            ratios = [o.get('ratio') for o in open_orders if o.get('ratio') is not None]
            if not ratios:
                # If no open orders yet, we can still compute spread returns using pair_id
                # Try to get beta from any order or use default
                beta = 1.0  # Default beta if we can't determine
            else:
                beta = max(ratios)  # The larger ratio is always beta
            
            # Get symbols directly from pair_id (format: "APT-USDT_NEAR-USDT")
            pair_parts = pair_id.split('_')
            if len(pair_parts) != 2:
                logger.warning(f"Pair {pair_id} has invalid format, expected 'BASE1_BASE2'")
                continue
            
            # Extract base symbols from pair_id
            a_base = pair_parts[0].replace('-USDT', '')
            b_base = pair_parts[1].replace('-USDT', '')
            
            # Load historical data for both symbols
            
            # Get historical closes
            lookback_start = self.oms_client.current_time - timedelta(days=self.min_lookback_days)
            
            a_data = self.data_manager.load_data_period(
                a_base, '15m', 'index_ohlcv_futures',
                lookback_start, self.oms_client.current_time
            )
            b_data = self.data_manager.load_data_period(
                b_base, '15m', 'index_ohlcv_futures',
                lookback_start, self.oms_client.current_time
            )
            
            if a_data is None or b_data is None or len(a_data) == 0 or len(b_data) == 0:
                logger.warning(f"Insufficient data for pair {pair_id}")
                continue
            
            # Align on timestamp and compute spread
            a_data = a_data.set_index('timestamp') if 'timestamp' in a_data.columns else a_data
            b_data = b_data.set_index('timestamp') if 'timestamp' in b_data.columns else b_data
            
            # Resample to daily
            a_daily = a_data['close'].resample('1D').last().dropna()
            b_daily = b_data['close'].resample('1D').last().dropna()
            
            # Align
            aligned = pd.concat([a_daily.rename('a'), b_daily.rename('b')], axis=1).dropna()
            
            if len(aligned) < self.min_lookback_days:
                logger.warning(f"Insufficient aligned data for pair {pair_id}")
                continue
            
            # Compute spread: spread = Price_A - beta * Price_B
            aligned['spread'] = aligned['a'] - beta * aligned['b']
            
            # Compute spread returns (percentage change)
            spread_returns_series = aligned['spread'].pct_change().dropna()
            
            # Store for later use
            self.pair_spread_history[pair_id] = spread_returns_series
            
            # Store in dict for DataFrame construction
            spread_returns_dict[pair_id] = spread_returns_series
        
        if not spread_returns_dict:
            return pd.DataFrame()
        
        # Align all pairs on common dates
        spread_returns_df = pd.DataFrame(spread_returns_dict)
        spread_returns_df = spread_returns_df.dropna()
        
        return spread_returns_df
    
    def _size_pairs(self, 
                    pairs_dict: Dict[str, List[Dict[str, Any]]],
                    balance: float) -> Dict[str, float]:
        """
        Allocate capital to pairs using pyportfolioopt.
        
        Uses risk parity or other optimization methods to allocate capital
        such that each pair contributes equal risk (or optimizes for min vol/max sharpe).
        """
        # Compute spread returns for all pairs
        spread_returns = self._compute_spread_returns(pairs_dict)
        
        if spread_returns.empty or len(spread_returns) < self.min_lookback_days:
            logger.warning("Insufficient spread returns data, using equal allocation")
        
        # Calculate expected returns and covariance matrix
        try:
            mu = expected_returns.mean_historical_return(spread_returns)
            S = risk_models.sample_cov(spread_returns)
            
            if self.risk_method == 'min_volatility':
                ef = EfficientFrontier(mu, S)
                weights = ef.min_volatility()
            elif self.risk_method == 'max_sharpe':
                ef = EfficientFrontier(mu, S)
                weights = ef.max_sharpe()
            else:
                logger.warning(f"Unknown risk_method {self.risk_method}")
            
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            # Allocate capital
            total_capital = balance * self.portfolio_alloc_frac
            allocations = {pair_id: total_capital * weight 
                          for pair_id, weight in weights.items()}
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}", exc_info=True)
            return None

    def _size_orders(self, 
                       pairs_dict: Dict[str, List[Dict[str, Any]]],
                       pair_allocations: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Size each pair's legs using ratio field to ensure net beta neutrality.
        
        Uses the ratio field directly:
        - Each order has a ratio (1 for one leg, beta for the other)
        - Total ratio = sum of ratios
        - Allocate: value = pair_capital * (ratio / total_ratio)
        
        This ensures proper hedge ratio: Notional_A / Notional_B = ratio_A / ratio_B = beta
        """
        sized_orders = []
        
        for pair_id, orders in pairs_dict.items():
            if pair_id not in pair_allocations:
                continue
            
            pair_capital = pair_allocations[pair_id]
            if pair_capital <= 0:
                continue
            
            # Filter out CLOSE orders
            open_orders = [o for o in orders if o.get('side') != 'CLOSE' and o.get('ratio') is not None]
            
            if len(open_orders) < 2:
                logger.warning(f"Pair {pair_id} has insufficient open orders for sizing")
                continue
            
            # Extract ratios
            ratios = [o.get('ratio') for o in open_orders if o.get('ratio') is not None]
            total_ratio = sum(ratios)
            
            if total_ratio <= 0:
                logger.warning(f"Pair {pair_id} has invalid total ratio: {total_ratio}")
                continue
            
            # Size each leg proportionally to its ratio
            for order in open_orders:
                ratio = order.get('ratio', 0)
                if ratio <= 0:
                    continue
                
                value = pair_capital * (ratio / total_ratio)
                sized_order = {**order, 'value': value}
                sized_orders.append(sized_order)
            
            # Log sizing info
            beta = max(ratios)  # The larger ratio is beta
            logger.debug(f"Pair {pair_id}: allocated {pair_capital:.2f} USDT, "
                        f"ratios={ratios}, beta={beta:.3f}")
        
        return sized_orders

