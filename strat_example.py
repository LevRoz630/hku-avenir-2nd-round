def run_backtest(single_trading_data: pd.DataFrame, single_minute_data: pd.DataFrame):
        trading_positions = pd.DataFrame(0.0, index=single_trading_data.index, columns=coins)
        trades = []
        calculated_betas = []

        for t in range(lookback_window, len(single_trading_data)):
            try:
                hist_data = single_trading_data.iloc[max(0, t - lookback_window):t]

                from sklearn.linear_model import LinearRegression
                log_prices = np.log(hist_data.values)
                y, x = log_prices[:, 0], log_prices[:, 1]

                reg = LinearRegression().fit(x.reshape(-1, 1), y)
                dynamic_beta = reg.coef_[0]
                calculated_betas.append(dynamic_beta)

                current_prices = single_trading_data.iloc[t].values
                current_spread = current_prices[0] - dynamic_beta * current_prices[1]

                historical_spreads = []
                for i in range(max(0, t - lookback_window), t):
                    hist_prices = single_trading_data.iloc[i].values
                    hist_spread = hist_prices[0] - dynamic_beta * hist_prices[1]
                    historical_spreads.append(hist_spread)

                spread_mean = np.mean(historical_spreads)
                spread_std = np.std(historical_spreads)

                if spread_std > 0:
                    z_score = (current_spread - spread_mean) / spread_std
                    current_position = trading_positions.iloc[t - 1].copy()

                    if abs(z_score) > entry_threshold and abs(current_position.sum()) < 0.1:
                        scale = min(abs(z_score) / entry_threshold, 2.0) * max_position

                        if z_score > entry_threshold:
                            new_position = pd.Series([-scale, dynamic_beta * scale], index=coins)
                        else:
                            new_position = pd.Series([scale, -dynamic_beta * scale], index=coins)

                        trading_positions.iloc[t] = new_position
                        trades.append({
                            'timestamp': single_trading_data.index[t],
                            'action': 'enter',
                            'positions': dict(zip(coins, new_position))
                        })

                    elif abs(z_score) < exit_threshold and abs(current_position.sum()) > 0.05:
                        trading_positions.iloc[t] = 0.0
                        trades.append({
                            'timestamp': single_trading_data.index[t],
                            'action': 'exit',
                        })

                    elif abs(z_score) > 3 * entry_threshold and abs(current_position.sum()) > 0.05:
                        trading_positions.iloc[t] = 0.0
                        trades.append({
                            'timestamp': single_trading_data.index[t],
                            'action': 'stop_loss',
                        })

                    else:
                        trading_positions.iloc[t] = current_position

            except Exception:
                trading_positions.iloc[t] = trading_positions.iloc[t - 1] if t > 0 else 0.0
                continue

        minute_positions = trading_positions.reindex(single_minute_data.index, method='ffill').fillna(0)

        minute_pnl = pd.Series(0.0, index=single_minute_data.index)

        for t in range(1, len(single_minute_data)):
            try:
                price_change = (single_minute_data.iloc[t] / single_minute_data.iloc[t - 1] - 1).values
                position_pnl = np.dot(minute_positions.iloc[t - 1].values, price_change)
                minute_pnl.iloc[t] = position_pnl
            except Exception:
                continue

        cumulative_pnl = minute_pnl.cumsum()
        total_return = cumulative_pnl.iloc[-1]

        running_max = cumulative_pnl.expanding().max()
        drawdown_series = cumulative_pnl - running_max
        max_drawdown = drawdown_series.min()

        win_trades = 0
        total_trades = 0
        for i in range(1, len(trades), 2):
            if i < len(trades) and trades[i]['action'] in ['exit', 'stop_loss']:
                entry_time = trades[i - 1]['timestamp']
                exit_time = trades[i]['timestamp']
                entry_pnl = cumulative_pnl.loc[entry_time] if entry_time in cumulative_pnl.index else 0
                exit_pnl = cumulative_pnl.loc[exit_time] if exit_time in cumulative_pnl.index else 0
                if exit_pnl > entry_pnl:
                    win_trades += 1
                total_trades += 1

        win_rate = win_trades / max(1, total_trades)
        sharpe_ratio = minute_pnl.mean() / minute_pnl.std() * np.sqrt(365 * 96) if minute_pnl.std() > 0 else 0

        results = {
            'coins': coins,
            'beta_deviations': np.mean(calculated_betas) / np.std(calculated_betas) if np.std(calculated_betas) > 0 else 0.0,
            'beta_series': pd.Series(calculated_betas, index=single_trading_data.index[lookback_window:]),
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'trading_positions': trading_positions,
            'minute_pnl': cumulative_pnl,
            'drawdown_series': drawdown_series,
            'trades': trades
        }
        return results