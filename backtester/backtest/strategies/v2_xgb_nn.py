from typing import List, Dict, Any
import numpy as np
import pandas as pd
from oms_simulation import OMSClient
from hist_data import HistoricalDataCollector
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import torch
    import torch.nn as nn
    import torch.optim as optim
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class XGBNNResidualStrategy:
    def __init__(self, symbols: List[str], historical_data_dir: str, lookback_days: int = 30,
                 prediction_horizon: int = 24, retrain_frequency: int = 7):
        self.symbols = symbols
        self.oms_client = None
        self.data_manager = None
        self.lookback_days = lookback_days
        self.prediction_horizon = prediction_horizon
        self.retrain_frequency = retrain_frequency
        
        self.xgb_models = {}
        self.nn_models = {}
        self.scalers = {}
        self.last_retrain_day = None
        
        if not XGB_AVAILABLE:
            raise ImportError("xgboost, sklearn, and torch are required for this strategy")
    
    def _get_price_data(self, base_symbol: str, hours: int = None) -> pd.DataFrame:
        dm = self.oms_client.data_manager
        symbol_key = f"{base_symbol}-USDT"
        df = dm.perpetual_index_ohlcv_data.get(symbol_key)
        if df is None or df.empty:
            df = dm.perpetual_index_ohlcv_data.get(base_symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        
        current_time_tz = pd.Timestamp(self.oms_client.current_time)
        if current_time_tz.tz is None:
            current_time_tz = current_time_tz.tz_localize('UTC')
        else:
            current_time_tz = current_time_tz.tz_convert('UTC')
        
        df = df[df['timestamp'] < current_time_tz]
        
        if hours is not None and hours > 0:
            window_start = current_time_tz - pd.Timedelta(hours=hours)
            df = df[df['timestamp'] >= window_start]
        
        df = df.sort_values('timestamp')
        return df
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 20:
            return pd.DataFrame()
        
        df = df.copy()
        df = df.set_index('timestamp')
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        features = pd.DataFrame(index=df.index)
        
        features['returns_1'] = close.pct_change(1)
        features['returns_4'] = close.pct_change(4)
        features['returns_24'] = close.pct_change(24)
        
        features['ma_5'] = close.rolling(5).mean() / close - 1
        features['ma_10'] = close.rolling(10).mean() / close - 1
        features['ma_20'] = close.rolling(20).mean() / close - 1
        
        features['std_5'] = close.rolling(5).std() / close
        features['std_20'] = close.rolling(20).std() / close
        
        features['rsi'] = self._calculate_rsi(close, 14)
        
        features['high_low_ratio'] = (high - low) / close
        
        features['volume_ma_ratio'] = volume / volume.rolling(20).mean()
        
        target_values = close.shift(-self.prediction_horizon) / close - 1
        features['target'] = target_values
        
        features = features.dropna()
        
        if len(features) < 20:
            return pd.DataFrame()
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100
    
    def _train_xgb_model(self, X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1
        )
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model
    
    class ResidualNN(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 16)
            self.fc4 = nn.Linear(16, 1)
            self.dropout = nn.Dropout(0.2)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    
    def _train_nn_model(self, X: np.ndarray, residuals: np.ndarray) -> nn.Module:
        X_train, X_val, r_train, r_val = train_test_split(X, residuals, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = self.ResidualNN(X_train_scaled.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        r_train_tensor = torch.FloatTensor(r_train.reshape(-1, 1))
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        r_val_tensor = torch.FloatTensor(r_val.reshape(-1, 1))
        
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, r_train_tensor)
            loss.backward()
            optimizer.step()
        
        model.eval()
        return model, scaler
    
    def _should_retrain(self) -> bool:
        if self.last_retrain_day is None:
            return True
        
        current_day = (self.oms_client.current_time.year, 
                      self.oms_client.current_time.month, 
                      self.oms_client.current_time.day)
        
        return current_day != self.last_retrain_day
    
    def _retrain_models(self):
        current_day = (self.oms_client.current_time.year,
                      self.oms_client.current_time.month,
                      self.oms_client.current_time.day)
        
        logger.info(f"Retraining models for {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            base_symbol = symbol.replace('-PERP', '').replace('-USDT', '')
            
            try:
                df = self._get_price_data(base_symbol, hours=self.lookback_days * 24)
                if df.empty or len(df) < 100:
                    full_df = self.oms_client.data_manager.perpetual_index_ohlcv_data.get(f"{base_symbol}-USDT") or self.oms_client.data_manager.perpetual_index_ohlcv_data.get(base_symbol)
                    if full_df is not None and not full_df.empty:
                        logger.warning(f"Insufficient data for {base_symbol}: {len(df)} rows after filtering, {len(full_df)} total rows, current_time={self.oms_client.current_time}, first_ts={full_df['timestamp'].min()}, last_ts={full_df['timestamp'].max()}")
                    else:
                        logger.warning(f"No data loaded for {base_symbol}")
                    continue
                
                features = self._create_features(df)
                if features.empty or len(features) < 50:
                    logger.warning(f"Insufficient features for {base_symbol}: {len(features)} rows from {len(df)} data rows")
                    continue
                
                feature_cols = [c for c in features.columns if c != 'target']
                X = features[feature_cols].values
                y = features['target'].values
                
                train_size = len(X) - max(self.prediction_horizon, 5)
                if train_size < 30:
                    logger.warning(f"Insufficient training data for {base_symbol}: {train_size} samples from {len(features)} features")
                    continue
                
                X_train = X[:train_size]
                y_train = y[:train_size]
                
                xgb_model = self._train_xgb_model(X_train, y_train)
                xgb_pred = xgb_model.predict(X_train)
                residuals = y_train - xgb_pred
                
                nn_model, scaler = self._train_nn_model(X_train, residuals)
                
                self.xgb_models[base_symbol] = xgb_model
                self.nn_models[base_symbol] = nn_model
                self.scalers[base_symbol] = scaler
                
                logger.info(f"Trained models for {base_symbol}: XGB R2={np.corrcoef(y_train, xgb_pred)[0,1]**2:.3f}, Residual std={np.std(residuals):.4f}")
            except Exception as e:
                logger.error(f"Error training models for {base_symbol}: {e}")
                continue
        
        self.last_retrain_day = current_day
        logger.info(f"Training complete. Models available: {list(self.xgb_models.keys())}")
    
    def _predict_return(self, base_symbol: str) -> float:
        if base_symbol not in self.xgb_models:
            return 0.0
        
        try:
            df = self._get_price_data(base_symbol, hours=48)
            if df.empty:
                return 0.0
            
            features = self._create_features(df)
            if features.empty or len(features) == 0:
                return 0.0
            
            feature_cols = [c for c in features.columns if c != 'target']
            if len(feature_cols) == 0:
                return 0.0
            
            X_latest = features[feature_cols].iloc[-1:].values
            
            xgb_model = self.xgb_models[base_symbol]
            xgb_pred = xgb_model.predict(X_latest)[0]
            
            nn_model = self.nn_models[base_symbol]
            scaler = self.scalers[base_symbol]
            X_scaled = scaler.transform(X_latest)
            X_tensor = torch.FloatTensor(X_scaled)
            
            nn_model.eval()
            with torch.no_grad():
                nn_residual = nn_model(X_tensor).item()
            
            combined_pred = xgb_pred + nn_residual
            return float(combined_pred)
        except Exception as e:
            logger.error(f"Error predicting return for {base_symbol}: {e}")
            return 0.0
    
    def run_strategy(self, oms_client: OMSClient, data_manager: HistoricalDataCollector):
        self.oms_client = oms_client
        self.data_manager = data_manager
        
        if self._should_retrain():
            try:
                self._retrain_models()
            except Exception as e:
                logger.error(f"Error retraining models: {e}")
                return []
        
        if not self.xgb_models:
            return []
        
        orders = []
        
        for symbol in self.symbols:
            base_symbol = symbol.replace('-PERP', '').replace('-USDT', '')
            if base_symbol not in self.xgb_models:
                continue
            
            predicted_return = self._predict_return(base_symbol)
            perp_symbol = symbol + '-PERP' if not symbol.endswith('-PERP') else symbol
            
            if abs(predicted_return) > 0.001:
                side = 'LONG' if predicted_return > 0 else 'SHORT'
                alloc_frac = min(abs(predicted_return) * 10, 0.15)
                orders.append({
                    'symbol': perp_symbol,
                    'instrument_type': 'future',
                    'side': side,
                    'alloc_frac': alloc_frac
                })
        
        return orders

