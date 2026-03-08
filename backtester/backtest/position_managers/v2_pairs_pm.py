from typing import List, Dict, Any, Optional
import logging
from datetime import timedelta
from src.hist_data import HistoricalDataCollector
from src.oms_simulation import OMSClient
import pandas as pd
from pypfopt import risk_models, expected_returns, EfficientFrontier

logger = logging.getLogger(__name__)
class PositionManager:
    def __init__(self,
                 portfolio_alloc_frac: float = 0.8,
                 risk_method: str = 'min_volatility',
                 min_lookback_days: int = 90):
        self.oms_client = None
        self.data_manager = None
        self.portfolio_alloc_frac = portfolio_alloc_frac
        self.risk_method = risk_method
        self.min_lookback_days = min_lookback_days
        self.pair_spread_history = {}  # Track spread returns for each pair

    def filter_orders(self, orders: List[Dict[str, Any]],
                     oms_client: OMSClient,
                     data_manager: HistoricalDataCollector) -> Optional[List[Dict[str, Any]]]:
        self.oms_client = oms_client
        self.data_manager = data_manager
        self.pairs_dict = self._group_orders_by_pair(orders)

        try:
            close_orders = [o for o in orders if o.get('side') == 'CLOSE']
            open_orders = [o for o in orders if o.get('side') != 'CLOSE']

            if not open_orders:
                return close_orders if close_orders else None

            # Allocate capital to pairs based on risk management
            pair_allocations = self._size_pairs(self.pairs_dict, oms_client.balance['USDT'])

            if not pair_allocations:
                logger.warning("No capital allocated to any pairs")
                return close_orders if close_orders else None

            # Size the orders for each leg based on beta
            sized_orders = self._size_orders(self.pairs_dict, pair_allocations)

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
        spread_returns_dict = {}

        for pair_id, orders in pairs_dict.items():
            open_orders = [o for o in orders if o.get('side') != 'CLOSE' and o.get('ratio') is not None]

            if len(open_orders) < 2:
                logger.warning(f"Pair {pair_id} has insufficient open orders")
                continue

            ratios = [o.get('ratio') for o in open_orders if o.get('ratio') is not None]
            if not ratios:
                beta = 1.0
            else:
                beta = max(ratios)

            # Get symbols from pair_id (format: "APT-USDT_NEAR-USDT")
            pair_parts = pair_id.split('_')
            if len(pair_parts) != 2:
                logger.warning(f"Pair {pair_id} has invalid format, expected 'BASE1_BASE2'")
                continue

            a_base = pair_parts[0].replace('-USDT', '')
            b_base = pair_parts[1].replace('-USDT', '')

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

            a_data = a_data.set_index('timestamp') if 'timestamp' in a_data.columns else a_data
            b_data = b_data.set_index('timestamp') if 'timestamp' in b_data.columns else b_data

            a_daily = a_data['close'].resample('1D').last().dropna()
            b_daily = b_data['close'].resample('1D').last().dropna()

            aligned = pd.concat([a_daily.rename('a'), b_daily.rename('b')], axis=1).dropna()

            if len(aligned) < self.min_lookback_days:
                logger.warning(f"Insufficient aligned data for pair {pair_id}")
                continue

            aligned['spread'] = aligned['a'] - beta * aligned['b']
            spread_returns_series = aligned['spread'].pct_change().dropna()

            self.pair_spread_history[pair_id] = spread_returns_series
            spread_returns_dict[pair_id] = spread_returns_series

        if not spread_returns_dict:
            return pd.DataFrame()

        spread_returns_df = pd.DataFrame(spread_returns_dict)
        spread_returns_df = spread_returns_df.dropna()

        return spread_returns_df

    def _size_pairs(self,
                    pairs_dict: Dict[str, List[Dict[str, Any]]],
                    balance: float) -> Dict[str, float]:
        spread_returns = self._compute_spread_returns(pairs_dict)

        if spread_returns.empty or len(spread_returns) < self.min_lookback_days:
            logger.warning("Insufficient spread returns data, using equal allocation")

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

            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}

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
        """Size legs using ratio field to maintain hedge ratio."""
        sized_orders = []

        for pair_id, orders in pairs_dict.items():
            if pair_id not in pair_allocations:
                continue

            pair_capital = pair_allocations[pair_id]
            if pair_capital <= 0:
                continue

            open_orders = [o for o in orders if o.get('side') != 'CLOSE' and o.get('ratio') is not None]

            if len(open_orders) < 2:
                logger.warning(f"Pair {pair_id} has insufficient open orders for sizing")
                continue

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
                sized_order = {**order, 'value': value}
                sized_orders.append(sized_order)

            beta = max(ratios)
            logger.debug(f"Pair {pair_id}: allocated {pair_capital:.2f} USDT, "
                        f"ratios={ratios}, beta={beta:.3f}")

        return sized_orders
