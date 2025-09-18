from datetime import datetime, timedelta
from typing import List, Dict, Any
from oms_simulation import BacktesterOMS
import logging
import numpy as np
from hist_data import HistoricalDataManager

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, historical_data_dir: str = "../historical_data"):
        self.historical_data_dir = historical_data_dir
        self.current_time = None
        self.start_time = None
        self.end_time = None
        self.oms_client = BacktesterOMS(historical_data_dir=historical_data_dir)
        self.data_manager = HistoricalDataManager(data_dir=historical_data_dir)
        self.portfolio_values = []
        self.daily_returns = []
        self.max_drawdown = 0
        self.sharpe_ratio = 0
        self.trade_history = []
        self.final_balance = 0
        self.final_positions = []

    def run_backtest(self, 
                        strategy_class: Any,
                        symbols: List[str],
                        start_date: datetime = None,
                        end_date: datetime = None,
                        time_step: timedelta = None) -> Dict[str, Any]:
        """
        Run backtest with a strategy class
        
        Args:
            strategy_class: Strategy class to test
            start_date: Start date for backtest
            end_date: End date for backtest
            time_step: Time step for backtest iteration
            
        Returns:
            Dictionary with backtest results and performance metrics
        """
        self.oms_client.set_data_manager(self.data_manager)
                # Pass on all the attributes of the class to the trategy for execution
        strategy = strategy_class(symbols=symbols, oms_client=self.oms_client)
        self.start_time = start_date
        self.end_time = end_date
        self.current_time = start_date
        if time_step is None:
            time_step = timedelta(hours=1)
        # Run backtest
        iteration = 0

        while self.current_time <= end_date:
            try:
                # Update OMS client's current time
                self.oms_client.set_current_time(self.current_time)

                # Update strategy's current time
                self.current_time = self.current_time
                
                total_value = self.oms_client.get_total_portfolio_value()
                print(f"debug total value:{total_value}")
                self.portfolio_values.append(total_value)
                summary = self.oms_client.get_position_summary()
                logger.info(f"Total Portfolio Value: {total_value}")
                logger.info(f"Position Summary: {summary}")

                strategy.run_strategy()
                
                
                # Move to next time step
                self.current_time += time_step
                iteration += 1
                
                # Log progress every 24 iterations (daily if hourly steps)
                if iteration % 24 == 0:
                    logger.info(f"Backtest progress: {self.current_time} (Iteration {iteration})")
                    
            except Exception as e:
                logger.error(f"Error in backtest iteration {iteration}: {e}")
                self.current_time += time_step
                continue
        
        # Calculate final performance metrics
        self.calculate_performance_metrics()
        
        # Return results
        return {
            'portfolio_values': self.portfolio_values,
            'daily_returns': self.daily_returns,
            'total_return': (self.portfolio_values[-1] / self.portfolio_values[0] - 1) if self.portfolio_values else 0,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'trade_history': self.trade_history,
            'final_balance': self.oms_client.get_account_balance(),
            'final_positions': self.oms_client.get_position()
        }

    def calculate_performance_metrics(self):
        """Calculate performance metrics"""
        if len(self.portfolio_values) < 2:
            return
            
        # Calculate daily returns
        self.period_returns = []
        for i in range(1, len(self.portfolio_values)):
            period_return = (self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1] #
            self.period_returns.append(period_return)
        
        # Calculate max drawdown
        peak = self.portfolio_values[0]
        self.max_drawdown = 0
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        
        # Calculate Sharpe ratio (simplified)
        if self.daily_returns:
            mean_return = np.mean(self.daily_returns)
            std_return = np.std(self.daily_returns)
            if std_return > 0:
                self.sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized

    def print_results(self, results: Dict[str, Any]):
        """Print backtest results in a formatted way"""
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Final Balance: {results['final_balance']}")
        print(f"Number of Trades: {len(results['trade_history'])}")
        print("="*50)

