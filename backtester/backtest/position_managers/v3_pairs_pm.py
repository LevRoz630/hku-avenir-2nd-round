from typing import List, Dict, Any, Optional
import logging
from datetime import timedelta
from src.hist_data import HistoricalDataCollector
from src.oms_simulation import OMSClient
import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns, EfficientFrontier

logger = logging.getLogger(__name__)
class PositionManager:
    """
    Pair-level position manager for pairs trading with rebalancing.

    Manages risk at the pair level, not individual symbols:
    1. Groups orders by pair_id
    2. Uses pyportfolioopt for max Sharpe ratio optimization
    3. Allocates capital to pairs based on risk-adjusted strategy returns
    4. Sizes each pair's legs using beta to ensure net beta neutrality
    5. Rebalances existing positions to optimal allocations
    6. Uses fixed max_total_allocation for all pairs
    """
    
    def __init__(self,
                 risk_method: str = 'max_sharpe',
                 min_lookback_days: int = 30,
                 rebalance_threshold: float = 0.05,
                 pairs_config: Optional[List[Dict[str, Any]]] = None,
                 max_total_allocation: float = 500.0):
        """
        Args:
            risk_method: 'min_volatility' or 'max_sharpe'
            min_lookback_days: Days of spread returns needed for risk calculation (default 30)
            rebalance_threshold: Minimum percentage drift before rebalancing (default 0.05 = 5%)
            pairs_config: Optional list of pair configs to build registry. If None, inferred from orders.
            max_total_allocation: Maximum total capital to allocate across all pairs (default $500)
        """
        self.oms_client = None
        self.data_manager = None
        self.risk_method = 'max_sharpe'  # Fixed to Sharpe ratio optimization
        self.min_lookback_days = min_lookback_days
        self.rebalance_threshold = rebalance_threshold
        self.max_total_allocation = max_total_allocation
        self.pair_spread_history = {}  # Track spread returns for each pair
        
        # Pair registry: {pair_id: {'a_symbol': str, 'b_symbol': str, 'a_base': str, 'b_base': str}}
        self.pair_registry = {}
        if pairs_config:
            self._build_pair_registry(pairs_config)
        
        # State tracking: {pair_id: {'current_side': str, 'last_beta': float, 'last_allocation': float}}
        self.pair_state = {}
        
    def filter_orders(self, orders: List[Dict[str, Any]], 
                     oms_client: OMSClient, 
                     data_manager: HistoricalDataCollector) -> Optional[List[Dict[str, Any]]]:
        """
        Main entry point for filtering and sizing orders with rebalancing.
        
        Steps:
        1. Separate CLOSE orders (strategy CLOSE takes precedence)
        2. Get existing positions and map to pairs
        3. Build pair registry from orders if not already built
        4. Filter out repeated signals for pairs with existing positions (unless rebalancing needed)
        5. Calculate optimal allocations for ALL pairs (existing + new orders)
        6. Calculate current allocations for existing pairs
        7. Generate rebalancing orders if drift exceeds threshold
        8. Size new orders based on optimal allocations
        9. Return all orders (close + rebalance + new)
        """
        self.oms_client = oms_client
        self.data_manager = data_manager
        
        # Log timestamp and incoming orders
        logger.info(f"\n{'='*80}")
        logger.info(f"POSITION MANAGER - Timestamp: {oms_client.current_time}")
        logger.info(f"{'='*80}")
        logger.info(f"Incoming orders: {len(orders)} total")
        
        try:
            # 1. Strategy CLOSE orders take precedence - pass through immediately
            close_orders = [o for o in orders if o.get('side') == 'CLOSE']
            open_orders = [o for o in orders if o.get('side') != 'CLOSE']
            
            logger.info(f"  - CLOSE orders: {len(close_orders)}")
            logger.info(f"  - OPEN orders: {len(open_orders)}")
            
            # 2. Build pair registry from orders if not already initialized
            if not self.pair_registry:
                self._build_pair_registry_from_orders(open_orders)
            
            # 3. Get existing positions and map to pairs
            existing_pair_positions = self._get_current_pair_positions()
            
            logger.info(f"Existing positions: {len(existing_pair_positions)} pairs")
            for pair_id, pair_data in existing_pair_positions.items():
                symbols = pair_data.get('symbols', [])
                logger.info(f"  - {pair_id}: {len(symbols)} legs")
            
            # Early return if no orders and no existing positions
            if not open_orders and not existing_pair_positions:
                logger.info("No orders and no positions - returning early")
                return close_orders if close_orders else None
            
            # 4. Filter out repeated signals for pairs with existing positions
            # (unless they need rebalancing - we'll check that later)
            filtered_open_orders = []
            for order in open_orders:
                pair_id = order.get('pair_id')
                if pair_id and pair_id in existing_pair_positions:
                    # Check if this pair needs rebalancing (will be checked later)
                    # For now, keep the order - we'll decide in rebalancing logic
                    filtered_open_orders.append(order)
                else:
                    filtered_open_orders.append(order)
            
            # 5. Group new orders by pair_id
            new_pairs_dict = self._group_orders_by_pair(filtered_open_orders)
            
            # 6. Merge existing pairs with new orders for full portfolio optimization
            all_pairs_dict = self._merge_existing_and_new_pairs(existing_pair_positions, new_pairs_dict)
            
            # 7. Calculate total portfolio value (capital base)
            total_portfolio_value = oms_client.update_portfolio_value()
            logger.info(f"Total Portfolio Value: ${total_portfolio_value:,.2f}")
            
            # 8. Calculate optimal allocations for ALL pairs (capped at max_total_allocation)
            optimal_allocations = self._size_pairs(all_pairs_dict, total_portfolio_value)
            
            if not optimal_allocations:
                logger.warning("No capital allocated to any pairs")
                return close_orders if close_orders else None
            
            # Log optimal allocations
            logger.info(f"\nOPTIMAL ALLOCATIONS (from {self.risk_method}):")
            total_optimal = sum(optimal_allocations.values())
            for pair_id, alloc in sorted(optimal_allocations.items(), key=lambda x: x[1], reverse=True):
                pct = (alloc / total_optimal * 100) if total_optimal > 0 else 0
                logger.info(f"  {pair_id}: ${alloc:,.2f} ({pct:.1f}%)")
            
            # 9. Calculate current allocations for existing pairs
            current_allocations = self._calculate_current_pair_allocations(existing_pair_positions)
            
            # Log current allocations
            logger.info(f"\nCURRENT ALLOCATIONS:")
            if current_allocations:
                total_current = sum(current_allocations.values())
                for pair_id, alloc in sorted(current_allocations.items(), key=lambda x: x[1], reverse=True):
                    pct = (alloc / total_current * 100) if total_current > 0 else 0
                    logger.info(f"  {pair_id}: ${alloc:,.2f} ({pct:.1f}%)")
            else:
                logger.info("  (no positions)")
            
            # Log allocation drift
            logger.info(f"\nALLOCATION DRIFT:")
            for pair_id in set(list(optimal_allocations.keys()) + list(current_allocations.keys())):
                optimal = optimal_allocations.get(pair_id, 0)
                current = current_allocations.get(pair_id, 0)
                drift = optimal - current
                drift_pct = (drift / optimal * 100) if optimal > 0 else 0
                logger.info(f"  {pair_id}: drift=${drift:,.2f} ({drift_pct:+.1f}%)")
            
            # 10. Generate rebalancing orders (threshold-based)
            rebalance_orders = self._generate_rebalance_orders(
                existing_pair_positions,
                current_allocations,
                optimal_allocations,
                close_orders  # Pass to avoid conflicts with strategy CLOSE
            )
            
            logger.info(f"\nREBALANCING:")
            if rebalance_orders:
                logger.info(f"  Generating {len(rebalance_orders)} rebalance orders")
                for order in rebalance_orders:
                    logger.info(f"    {order.get('side')} {order.get('symbol')} - ${order.get('value', 0):,.2f}")
            else:
                logger.info("  No rebalancing needed")
            
            # 11. Filter out new orders for pairs that already have positions (unless rebalancing)
            # If a pair has existing position and no rebalance needed, ignore new orders
            final_new_orders = []
            for order in filtered_open_orders:
                pair_id = order.get('pair_id')
                if pair_id and pair_id in existing_pair_positions:
                    # Check if we're rebalancing this pair
                    pair_rebalancing = any(
                        o.get('pair_id') == pair_id 
                        for o in rebalance_orders
                    )
                    if not pair_rebalancing:
                        # Skip - pair already has position and no rebalancing needed
                        continue
                final_new_orders.append(order)
            
            # 12. Size new orders using optimal allocations
            final_new_pairs_dict = self._group_orders_by_pair(final_new_orders)
            sized_new_orders = self._size_orders(final_new_pairs_dict, optimal_allocations)

            logger.info(f"\nNEW TRADES:")
            if sized_new_orders:
                logger.info(f"  Generating {len(sized_new_orders)} new trade orders")
                for order in sized_new_orders:
                    logger.info(f"    {order.get('side')} {order.get('symbol')} - ${order.get('value', 0):,.2f}")
            else:
                logger.info("  No new trades")

            # 13. Update state tracking
            self._update_pair_state(all_pairs_dict, optimal_allocations)
            
            # Summary
            final_orders = close_orders + rebalance_orders + sized_new_orders
            logger.info(f"\nFINAL SUMMARY:")
            logger.info(f"  Total orders to execute: {len(final_orders)}")
            logger.info(f"    - CLOSE: {len(close_orders)}")
            logger.info(f"    - REBALANCE: {len(rebalance_orders)}")
            logger.info(f"    - NEW TRADES: {len(sized_new_orders)}")
            logger.info(f"{'='*80}\n")
            
            return final_orders
            
        except Exception as e:
            logger.error(f"Error filtering orders: {e}", exc_info=True)
            return None
    
    def _build_pair_registry(self, pairs_config: List[Dict[str, Any]]) -> None:
        """Build pair registry from config."""
        for cfg in pairs_config:
            legs = cfg.get('legs', [])
            if len(legs) != 2:
                continue
            base_a, base_b = legs
            pair_id = f"{base_a}_{base_b}"
            a_base = base_a.replace('-USDT', '')
            b_base = base_b.replace('-USDT', '')
            self.pair_registry[pair_id] = {
                'a_symbol': f"{a_base}-PERP",
                'b_symbol': f"{b_base}-PERP",
                'a_base': a_base,
                'b_base': b_base
            }
    
    def _build_pair_registry_from_orders(self, orders: List[Dict[str, Any]]) -> None:
        """Build pair registry from orders by parsing pair_id."""
        for order in orders:
            pair_id = order.get('pair_id')
            if not pair_id or pair_id in self.pair_registry:
                continue
            
            # Parse pair_id format: "APT-USDT_NEAR-USDT"
            pair_parts = pair_id.split('_')
            if len(pair_parts) != 2:
                continue
            
            a_base = pair_parts[0].replace('-USDT', '')
            b_base = pair_parts[1].replace('-USDT', '')
            self.pair_registry[pair_id] = {
                'a_symbol': f"{a_base}-PERP",
                'b_symbol': f"{b_base}-PERP",
                'a_base': a_base,
                'b_base': b_base
            }
    
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
    
    def _compute_strategy_returns(self, pairs_dict: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Compute strategy returns instead of spread returns.
        Uses historical simulation of entry/exit signals to get actual strategy performance.
        """
        strategy_returns_dict = {}

        for pair_id, orders in pairs_dict.items():
            # Get spread data for this pair
            spread_returns = self._compute_spread_returns({pair_id: orders})
            if spread_returns.empty or pair_id not in spread_returns.columns:
                continue

            spread_series = spread_returns[pair_id]

            # Simulate strategy returns based on entry/exit logic
            # Entry: |z-score| > 1.5, Exit: |z-score| < 0.5
            entry_threshold = 1.5
            exit_threshold = 0.5

            # Calculate rolling z-score
            rolling_mean = spread_series.rolling(window=min(100, len(spread_series))).mean()
            rolling_std = spread_series.rolling(window=min(100, len(spread_series))).std()
            zscore = (spread_series - rolling_mean) / rolling_std

            # Simulate strategy positions and returns
            position = 0  # 1 for long spread, -1 for short spread, 0 for flat
            strategy_returns = []

            for i in range(len(spread_series)):
                current_z = zscore.iloc[i] if i < len(zscore) else 0

                # Entry logic
                if position == 0:
                    if current_z >= entry_threshold:
                        position = -1  # Short spread
                    elif current_z <= -entry_threshold:
                        position = 1   # Long spread

                # Exit logic
                elif abs(current_z) <= exit_threshold:
                    position = 0  # Close position

                # Calculate return (spread return * position)
                spread_return = spread_series.iloc[i] if i > 0 else 0
                strategy_return = spread_return * position
                strategy_returns.append(strategy_return)

            strategy_returns_dict[pair_id] = pd.Series(strategy_returns, index=spread_series.index)

        if not strategy_returns_dict:
            return pd.DataFrame()

        # Align all strategies on common dates
        strategy_returns_df = pd.DataFrame(strategy_returns_dict)
        strategy_returns_df = strategy_returns_df.dropna()

        return strategy_returns_df

    def _compute_spread_returns(self, pairs_dict: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Compute historical spread returns for each pair/basket.
        Returns DataFrame with columns = pair_ids, rows = dates
        
        Supports both pairs (2 legs) and baskets (multiple legs).
        For baskets, uses eigenvector weights from ratio field to compute spread.
        """
        spread_returns_dict = {}
        
        for pair_id, orders in pairs_dict.items():
            # Filter out CLOSE orders
            open_orders = [o for o in orders if o.get('side') != 'CLOSE' and o.get('ratio') is not None]
            
            if len(open_orders) < 2:
                logger.warning(f"Pair/Basket {pair_id} has insufficient open orders")
                continue
            
            # Extract symbols and ratios from orders
            symbol_ratios = {}
            for order in open_orders:
                symbol = order.get('symbol')
                ratio = order.get('ratio')
                if symbol and ratio:
                    symbol_ratios[symbol] = ratio
            
            if not symbol_ratios:
                logger.warning(f"Pair/Basket {pair_id} has no valid symbol/ratio data")
                continue
            
            # Extract base symbols (remove -PERP suffix)
            symbols = list(symbol_ratios.keys())
            base_symbols = [s.replace('-PERP', '') for s in symbols]
            
            # Check if this is a basket (multiple legs) or pair (2 legs)
            is_basket = len(base_symbols) > 2
            
            # Load historical data for all symbols
            lookback_start = self.oms_client.current_time - timedelta(days=self.min_lookback_days)
            
            price_data = {}
            for base_sym in base_symbols:
                data = self.data_manager.load_data_period(
                    base_sym, '15m', 'index_ohlcv_futures',
                    lookback_start, self.oms_client.current_time
                )
                
                if data is None or len(data) == 0:
                    logger.warning(f"Insufficient data for {base_sym} in {pair_id}")
                    break
                
                data = data.set_index('timestamp') if 'timestamp' in data.columns else data
                # Use 15m data directly (no resampling) to match strategy timeframe
                price_series = data['close']
                price_data[base_sym] = price_series
            
            if len(price_data) != len(base_symbols):
                logger.warning(f"Insufficient data for pair/basket {pair_id}")
                continue
            
            # Align all price series (15m bars)
            aligned_df = pd.DataFrame(price_data).dropna()
            
            # Convert min_lookback_days to bars (assuming 15m data: 96 bars/day)
            bars_per_day = 96
            min_lookback_bars = self.min_lookback_days * bars_per_day
            
            if len(aligned_df) < min_lookback_bars:
                logger.warning(f"Insufficient aligned data for pair/basket {pair_id}: {len(aligned_df)} bars < {min_lookback_bars} required")
                continue
            
            # Compute spread
            if is_basket:
                # Multi-leg basket: reconstruct eigenvector weights from orders
                # Use eigenvector_sign if available, otherwise infer from LONG/SHORT
                weight_vec = np.zeros(len(base_symbols))
                
                for order in open_orders:
                    symbol = order.get('symbol')
                    base = symbol.replace('-PERP', '')
                    if base in base_symbols:
                        idx = base_symbols.index(base)
                        ratio = order.get('ratio', 0)
                        
                        # Use eigenvector_sign if available (more accurate)
                        eigenvector_sign = order.get('eigenvector_sign')
                        if eigenvector_sign is not None:
                            weight_vec[idx] = eigenvector_sign * ratio
                        else:
                            # Fallback: infer sign from order side
                            side = order.get('side')
                            sign = 1.0 if side == 'LONG' else -1.0
                            weight_vec[idx] = sign * ratio
                
                # Normalize by sum of absolute values (like eigenvector normalization)
                sum_abs = np.sum(np.abs(weight_vec))
                if sum_abs > 0:
                    weight_vec = weight_vec / sum_abs
                else:
                    logger.warning(f"Invalid weight vector for basket {pair_id}")
                    continue
                
                # Compute spread: spread = log_prices @ weight_vec (matches strategy)
                log_prices = np.log(aligned_df[base_symbols].values)
                spread = log_prices @ weight_vec
            else:
                # Pair: use simple spread calculation (A - beta * B)
                # Extract beta from ratios
                ratios_list = [symbol_ratios[s] for s in symbols]
                beta = max(ratios_list) if ratios_list else 1.0
                
                # Determine which symbol gets beta
                a_base = base_symbols[0]
                b_base = base_symbols[1]
                
                # Compute spread: spread = Price_A - beta * Price_B
                spread = aligned_df[a_base].values - beta * aligned_df[b_base].values
            
            # Compute spread returns
            if is_basket:
                # For baskets, spread is in log space, so use diff (change in log spread)
                spread_series = pd.Series(spread, index=aligned_df.index)
                spread_returns_series = spread_series.diff().dropna()
            else:
                # For pairs, use percentage change
                spread_series = pd.Series(spread, index=aligned_df.index)
                spread_returns_series = spread_series.pct_change().dropna()
            
            # Store for later use
            self.pair_spread_history[pair_id] = spread_returns_series
            
            # Store in dict for DataFrame construction
            spread_returns_dict[pair_id] = spread_returns_series
        
        if not spread_returns_dict:
            return pd.DataFrame()
        
        # Align all pairs/baskets on common dates
        spread_returns_df = pd.DataFrame(spread_returns_dict)
        spread_returns_df = spread_returns_df.dropna()
        
        return spread_returns_df
    
    def _size_pairs(self,
                    pairs_dict: Dict[str, List[Dict[str, Any]]],
                    total_portfolio_value: float) -> Dict[str, float]:
        """
        Allocate capital to pairs using max Sharpe ratio optimization.

        Uses simulated strategy performance (entry at |z|>1.5, exit at |z|<0.5)
        to maximize risk-adjusted returns across baskets.
        """
        # Compute strategy returns instead of spread returns
        strategy_returns = self._compute_strategy_returns(pairs_dict)
        
        # Compute strategy returns instead of spread returns
        strategy_returns = self._compute_strategy_returns(pairs_dict)

        if strategy_returns.empty or len(strategy_returns) < 10:  # Need minimum data
            logger.warning("Insufficient strategy returns data, using equal allocation")
            return self._equal_allocation(pairs_dict, total_portfolio_value)

        # Calculate expected returns and covariance matrix from strategy performance
        try:
            mu = expected_returns.mean_historical_return(strategy_returns)
            S = risk_models.sample_cov(strategy_returns)
            
            # Validate that we have data before optimizing
            if len(mu) == 0 or S.shape[0] == 0:
                logger.warning("Empty strategy covariance matrix, using equal allocation")
                return self._equal_allocation(pairs_dict, total_portfolio_value)
            
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe()
            
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            # Use max_total_allocation directly
            total_capital = self.max_total_allocation
            
            allocations = {pair_id: total_capital * weight 
                          for pair_id, weight in weights.items()}
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error in strategy-based portfolio optimization: {e}", exc_info=True)
            return self._equal_allocation(pairs_dict, total_portfolio_value)
    
    def _equal_allocation(self, pairs_dict: Dict[str, List[Dict[str, Any]]], total_portfolio_value: float) -> Dict[str, float]:
        """Fallback to equal allocation when optimization fails or data is insufficient."""
        n_pairs = len(pairs_dict)
        if n_pairs == 0:
            return {}
        
        # Use max_total_allocation directly
        total_capital = self.max_total_allocation
        
        per_pair_capital = total_capital / n_pairs
        
        allocations = {pair_id: per_pair_capital for pair_id in pairs_dict.keys()}
        return allocations

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
            
            for order in open_orders:
                ratio = order.get('ratio', 0)
                if ratio <= 0:
                    continue
                value = pair_capital * (ratio / total_ratio)
                sized_orders.append({**order, 'value': value})
        
        return sized_orders
    
    def _get_current_pair_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get existing positions grouped by pair_id.
        Returns: {pair_id: {'symbols': [symbol1, symbol2], 'positions': {symbol: pos_data}}}
        Handles partial positions (only one leg) - will close remaining leg (decision 14).
        """
        existing_pairs = {}
        
        # Build symbol to pair_id mapping from registry
        symbol_to_pair = {}
        for pair_id, pair_info in self.pair_registry.items():
            symbol_to_pair[pair_info['a_symbol']] = pair_id
            symbol_to_pair[pair_info['b_symbol']] = pair_id
        
        # Also parse from pair_id if symbol not in registry (fallback)
        for symbol, pos in self.oms_client.positions.items():
            if abs(pos.get('quantity', 0)) <= 0:
                continue
            
            # Try to find pair_id from registry
            pair_id = symbol_to_pair.get(symbol)
            
            # Fallback: parse from symbol if not in registry
            if not pair_id:
                base = symbol.replace('-PERP', '')
                for pid in self.pair_registry.keys():
                    if base in pid:
                        pair_id = pid
                        break
            
            # If still not found, try to infer from pair_id format in registry
            if not pair_id:
                # Try to find any pair_id containing this base
                for pid in list(self.pair_registry.keys()):
                    if base in pid:
                        pair_id = pid
                        break
            
            if not pair_id:
                logger.debug(f"Could not map symbol {symbol} to pair_id")
                continue
            
            if pair_id not in existing_pairs:
                existing_pairs[pair_id] = {
                    'symbols': [],
                    'positions': {}
                }
            
            existing_pairs[pair_id]['symbols'].append(symbol)
            existing_pairs[pair_id]['positions'][symbol] = pos
        
        # Handle partial positions: if only one leg exists, mark for cleanup
        partial_pairs = []
        for pair_id, pair_data in existing_pairs.items():
            symbols = pair_data.get('symbols', [])
            if len(symbols) == 1:
                partial_pairs.append(pair_id)
                logger.warning(f"Pair {pair_id} has only one leg ({symbols[0]}), will close remaining leg")
        
        return existing_pairs
    
    def _merge_existing_and_new_pairs(self, 
                                       existing_pair_positions: Dict[str, Dict[str, Any]],
                                       new_pairs_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Merge existing pair positions with new orders.
        Returns dict compatible with _compute_spread_returns format.
        """
        merged = {}
        
        # Add existing pairs (need to create dummy orders for spread calculation)
        for pair_id, pair_data in existing_pair_positions.items():
            if pair_id not in merged:
                merged[pair_id] = []
            
            # Create placeholder orders for existing positions to enable spread calculation
            # We'll use state or infer from positions
            symbols = pair_data.get('symbols', [])
            if len(symbols) >= 2:
                # Try to infer side and beta from positions
                a_symbol = symbols[0]
                b_symbol = symbols[1]
                a_pos = pair_data['positions'].get(a_symbol, {})
                b_pos = pair_data['positions'].get(b_symbol, {})
                
                # Infer side and beta from positions
                a_side = a_pos.get('side', 'LONG')
                b_side = b_pos.get('side', 'LONG')
                
                # Get beta from state or use default
                beta = self.pair_state.get(pair_id, {}).get('last_beta', 1.0)
                
                # Create placeholder orders for spread calculation
                # Format matches what strategy sends
                if a_side == 'LONG' and b_side == 'SHORT':
                    # Long spread: A LONG, B SHORT
                    merged[pair_id].append({
                        'symbol': a_symbol,
                        'instrument_type': 'future',
                        'side': 'LONG',
                        'pair_id': pair_id,
                        'ratio': beta
                    })
                    merged[pair_id].append({
                        'symbol': b_symbol,
                        'instrument_type': 'future',
                        'side': 'SHORT',
                        'pair_id': pair_id,
                        'ratio': 1
                    })
                elif a_side == 'SHORT' and b_side == 'LONG':
                    # Short spread: A SHORT, B LONG
                    merged[pair_id].append({
                        'symbol': a_symbol,
                        'instrument_type': 'future',
                        'side': 'SHORT',
                        'pair_id': pair_id,
                        'ratio': 1
                    })
                    merged[pair_id].append({
                        'symbol': b_symbol,
                        'instrument_type': 'future',
                        'side': 'LONG',
                        'pair_id': pair_id,
                        'ratio': beta
                    })
        
        # Add new orders
        for pair_id, orders in new_pairs_dict.items():
            if pair_id not in merged:
                merged[pair_id] = []
            merged[pair_id].extend(orders)
        
        return merged
    
    def _calculate_current_pair_allocations(self, 
                                            existing_pair_positions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate current notional value allocated to each pair.
        Returns: {pair_id: current_notional_value}
        """
        current_allocations = {}
        
        for pair_id, pair_data in existing_pair_positions.items():
            total_notional = 0.0
            positions = pair_data.get('positions', {})
            
            for symbol, pos in positions.items():
                current_price = self.oms_client.get_current_price(
                    symbol, 
                    pos.get('instrument_type', 'future')
                )
                if current_price:
                    notional = abs(pos.get('quantity', 0)) * current_price
                    total_notional += notional
            
            current_allocations[pair_id] = total_notional
        
        return current_allocations
    
    def _detect_position_side(self, pair_id: str, pair_data: Dict[str, Any]) -> Optional[str]:
        """
        Infer position side (long_spread or short_spread) from positions.
        Returns 'long_spread', 'short_spread', or None if ambiguous.
        """
        positions = pair_data.get('positions', {})
        symbols = pair_data.get('symbols', [])
        
        if len(symbols) < 2:
            return None
        
        a_symbol = symbols[0]
        b_symbol = symbols[1]
        a_pos = positions.get(a_symbol, {})
        b_pos = positions.get(b_symbol, {})
        
        a_side = a_pos.get('side')
        b_side = b_pos.get('side')
        
        if a_side == 'LONG' and b_side == 'SHORT':
            return 'long_spread'
        elif a_side == 'SHORT' and b_side == 'LONG':
            return 'short_spread'
        
        return None
    
    def _generate_rebalance_orders(self,
                                    existing_pair_positions: Dict[str, Dict[str, Any]],
                                    current_allocations: Dict[str, float],
                                    optimal_allocations: Dict[str, float],
                                    close_orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate orders to rebalance existing positions to optimal allocations.
        Only rebalances if drift exceeds threshold.
        Strategy CLOSE orders take precedence - skip rebalancing if strategy wants to close.
        """
        rebalance_orders = []
        
        # Get pairs that strategy wants to close (take precedence)
        strategy_close_pairs = set()
        for order in close_orders:
            pair_id = order.get('pair_id')
            if pair_id:
                strategy_close_pairs.add(pair_id)
        
        for pair_id, pair_data in existing_pair_positions.items():
            # Skip if strategy wants to close this pair
            if pair_id in strategy_close_pairs:
                logger.debug(f"Skipping rebalance for {pair_id} - strategy wants to close")
                continue
            
            # Handle partial positions (only one leg) - close remaining leg (decision 14)
            symbols = pair_data.get('symbols', [])
            if len(symbols) == 1:
                logger.info(f"Closing partial position for pair {pair_id} - only one leg exists")
                rebalance_orders.append({
                    'symbol': symbols[0],
                    'instrument_type': 'future',
                    'side': 'CLOSE',
                    'pair_id': pair_id
                })
                continue
            
            current_notional = current_allocations.get(pair_id, 0.0)
            optimal_notional = optimal_allocations.get(pair_id, 0.0)
            
            # If optimal is zero or negative, close the position
            if optimal_notional <= 0:
                logger.info(f"Closing pair {pair_id} - optimal allocation is zero")
                symbols = pair_data.get('symbols', [])
                for symbol in symbols:
                    rebalance_orders.append({
                        'symbol': symbol,
                        'instrument_type': 'future',
                        'side': 'CLOSE',
                        'pair_id': pair_id
                    })
                continue
            
            # Calculate drift percentage
            if current_notional <= 0:
                # No current position, but optimization says to open - handled by new orders
                continue
            
            drift_pct = abs(optimal_notional - current_notional) / optimal_notional
            
            # Only rebalance if drift exceeds threshold
            if drift_pct < self.rebalance_threshold:
                logger.debug(f"Pair {pair_id} drift {drift_pct:.2%} below threshold {self.rebalance_threshold:.2%}")
                continue
            
            logger.info(f"Rebalancing pair {pair_id}: current=${current_notional:.2f}, "
                       f"optimal=${optimal_notional:.2f}, drift={drift_pct:.2%}")
            
            # Detect current side
            current_side = self._detect_position_side(pair_id, pair_data)
            if not current_side:
                logger.warning(f"Could not detect side for pair {pair_id}, skipping rebalance")
                continue
            
            # Get symbols from registry or pair_data
            if pair_id in self.pair_registry:
                a_symbol = self.pair_registry[pair_id]['a_symbol']
                b_symbol = self.pair_registry[pair_id]['b_symbol']
            else:
                symbols = pair_data.get('symbols', [])
                if len(symbols) < 2:
                    continue
                a_symbol = symbols[0]
                b_symbol = symbols[1]
            
            # Get beta from state or use default
            beta = self.pair_state.get(pair_id, {}).get('last_beta', 1.0)
            
            # Calculate target notional for each leg (proportional scaling)
            # Maintain current beta by scaling both legs proportionally
            scale_factor = optimal_notional / current_notional
            
            # Calculate current notional per leg
            a_pos = pair_data['positions'].get(a_symbol, {})
            b_pos = pair_data['positions'].get(b_symbol, {})
            
            a_price = self.oms_client.get_current_price(a_symbol, 'future')
            b_price = self.oms_client.get_current_price(b_symbol, 'future')
            
            if not a_price or not b_price:
                logger.warning(f"Could not get prices for pair {pair_id}")
                continue
            
            a_current_notional = abs(a_pos.get('quantity', 0)) * a_price
            b_current_notional = abs(b_pos.get('quantity', 0)) * b_price
            
            # Target notional (scaled proportionally)
            a_target_notional = a_current_notional * scale_factor
            b_target_notional = b_current_notional * scale_factor
            
            # Calculate delta (additive orders)
            a_delta = a_target_notional - a_current_notional
            b_delta = b_target_notional - b_current_notional
            
            # Generate additive orders to reach target
            if abs(a_delta) > 0.01:  # Minimum threshold to avoid tiny orders
                a_side = 'LONG' if a_delta > 0 else 'SHORT'
                # If currently opposite side, need to flip
                if a_pos.get('side') and a_pos.get('side') != a_side:
                    # Close first, then open new
                    rebalance_orders.append({
                        'symbol': a_symbol,
                        'instrument_type': 'future',
                        'side': 'CLOSE',
                        'pair_id': pair_id
                    })
                    rebalance_orders.append({
                        'symbol': a_symbol,
                        'instrument_type': 'future',
                        'side': a_side,
                        'pair_id': pair_id,
                        'value': abs(a_target_notional),
                        'ratio': beta if current_side == 'long_spread' else 1
                    })
                else:
                    # Same side, additive
                    rebalance_orders.append({
                        'symbol': a_symbol,
                        'instrument_type': 'future',
                        'side': a_side,
                        'pair_id': pair_id,
                        'value': abs(a_delta),
                        'ratio': beta if current_side == 'long_spread' else 1
                    })
            
            if abs(b_delta) > 0.01:
                b_side = 'LONG' if b_delta > 0 else 'SHORT'
                if b_pos.get('side') and b_pos.get('side') != b_side:
                    rebalance_orders.append({
                        'symbol': b_symbol,
                        'instrument_type': 'future',
                        'side': 'CLOSE',
                        'pair_id': pair_id
                    })
                    rebalance_orders.append({
                        'symbol': b_symbol,
                        'instrument_type': 'future',
                        'side': b_side,
                        'pair_id': pair_id,
                        'value': abs(b_target_notional),
                        'ratio': 1 if current_side == 'long_spread' else beta
                    })
                else:
                    rebalance_orders.append({
                        'symbol': b_symbol,
                        'instrument_type': 'future',
                        'side': b_side,
                        'pair_id': pair_id,
                        'value': abs(b_delta),
                        'ratio': 1 if current_side == 'long_spread' else beta
                    })
        
        return rebalance_orders
    
    def _update_pair_state(self, 
                           pairs_dict: Dict[str, List[Dict[str, Any]]],
                           optimal_allocations: Dict[str, float]) -> None:
        """Update state tracking for pairs."""
        for pair_id, orders in pairs_dict.items():
            if pair_id not in self.pair_state:
                self.pair_state[pair_id] = {}
            
            # Extract beta from orders
            open_orders = [o for o in orders if o.get('side') != 'CLOSE' and o.get('ratio') is not None]
            if open_orders:
                ratios = [o.get('ratio') for o in open_orders if o.get('ratio') is not None]
                if ratios:
                    self.pair_state[pair_id]['last_beta'] = max(ratios)
            
            # Update allocation
            if pair_id in optimal_allocations:
                self.pair_state[pair_id]['last_allocation'] = optimal_allocations[pair_id]
            
            # Update side from orders
            if open_orders:
                # Infer side from order sides
                sides = [o.get('side') for o in open_orders]
                symbols = [o.get('symbol') for o in open_orders]
                if len(sides) == 2 and len(symbols) == 2:
                    # Check if we can determine side
                    if pair_id in self.pair_registry:
                        a_sym = self.pair_registry[pair_id]['a_symbol']
                        b_sym = self.pair_registry[pair_id]['b_symbol']
                        
                        a_order = next((o for o in open_orders if o['symbol'] == a_sym), None)
                        b_order = next((o for o in open_orders if o['symbol'] == b_sym), None)
                        
                        if a_order and b_order:
                            a_side = a_order.get('side')
                            b_side = b_order.get('side')
                            if a_side == 'LONG' and b_side == 'SHORT':
                                self.pair_state[pair_id]['current_side'] = 'long_spread'
                            elif a_side == 'SHORT' and b_side == 'LONG':
                                self.pair_state[pair_id]['current_side'] = 'short_spread'

