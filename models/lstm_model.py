# models/lstm_model.py
"""
Module for LSTM (Long Short-Term Memory) neural network model processing.
Updated to predict both buy and sell prices.
Handles TensorFlow/Keras dependencies gracefully.
Optimized for MAE ≈ 0.05 target.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import traceback
import joblib
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from datetime import timedelta

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.layers import Bidirectional, Attention, GlobalAveragePooling1D, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
    from tensorflow.keras.regularizers import l2
    from tqdm import tqdm
    LSTM_AVAILABLE = True
    tf.get_logger().setLevel('ERROR')
except ImportError:
    LSTM_AVAILABLE = False
    logging.warning("TensorFlow not found. LSTM models will be disabled.")
    tf = None

from utils.datetime_utils import convert_numpy_datetime64_to_python_datetime

# TQDM Progress bar callback
class TQDMProgressCallback(Callback):
    def __init__(self, epochs, verbose=1):
        super().__init__()
        self.epochs = epochs
        self.verbose = verbose
        self.pbar = None
        
    def on_train_begin(self, logs=None):
        if self.verbose:
            self.pbar = tqdm(total=self.epochs, desc="Training", leave=True)
        
    def on_epoch_end(self, epoch, logs=None):
        if self.verbose and self.pbar is not None:
            # Get metrics
            mae = logs.get('mae', 0)
            val_mae = logs.get('val_mae', 0)
            
            # Update progress bar
            self.pbar.set_postfix({
                'MAE': f'{mae:.6f}',
                'Val_MAE': f'{val_mae:.6f}'
            })
            self.pbar.update(1)
            
    def on_train_end(self, logs=None):
        if self.verbose and self.pbar is not None:
            self.pbar.close()

# Creates sequences for LSTM training with feature engineering
def create_sequences(data_dict, seq_length, feature_cols, target_col_idx=0):
    X, y = [], []
    
    # Get the length from the first feature
    data_length = len(data_dict[feature_cols[0]])
    
    for i in range(data_length - seq_length):
        sequence = []
        for col in feature_cols:
            sequence.append(data_dict[col][i:(i + seq_length)])
        
        X.append(np.column_stack(sequence))
        y.append(data_dict[feature_cols[target_col_idx]][i + seq_length])  # Target is configurable
    
    return np.array(X), np.array(y)

# Engineer features for better predictions
def engineer_features(df, price_col='buyprice'):
    df = df.with_columns([
        # Basic price features
        (pl.col(price_col).diff().fill_null(0)).alias(f'{price_col}_diff'),
        (pl.col(price_col).shift(1).fill_null(strategy="forward")).alias(f'{price_col}_lag1'),
        (pl.col(price_col).shift(2).fill_null(strategy="forward")).alias(f'{price_col}_lag2'),
        (pl.col(price_col).shift(3).fill_null(strategy="forward")).alias(f'{price_col}_lag3'),
        
        # Moving averages
        (pl.col(price_col).rolling_mean(3).fill_null(strategy="forward")).alias(f'{price_col}_ma3'),
        (pl.col(price_col).rolling_mean(5).fill_null(strategy="forward")).alias(f'{price_col}_ma5'),
        (pl.col(price_col).rolling_mean(7).fill_null(strategy="forward")).alias(f'{price_col}_ma7'),
        (pl.col(price_col).rolling_mean(12).fill_null(strategy="forward")).alias(f'{price_col}_ma12'),
        
        # Volatility measures
        (pl.col(price_col).rolling_std(3).fill_null(strategy="forward")).alias(f'{price_col}_std3'),
        (pl.col(price_col).rolling_std(7).fill_null(strategy="forward")).alias(f'{price_col}_std7'),
        
        # Momentum indicators
        (pl.col(price_col).diff(3).fill_null(0)).alias(f'{price_col}_momentum3'),
        (pl.col(price_col).diff(7).fill_null(0)).alias(f'{price_col}_momentum7'),
        
        # Range features
        (pl.col(price_col).rolling_max(3).fill_null(strategy="forward") - 
         pl.col(price_col).rolling_min(3).fill_null(strategy="forward")).alias(f'{price_col}_range3'),
        
        # Return features
        ((pl.col(price_col) / pl.col(price_col).shift(1) - 1).fill_null(0)).alias(f'{price_col}_return'),
        ((pl.col(price_col) / pl.col(price_col).shift(3) - 1).fill_null(0)).alias(f'{price_col}_return3'),
        
        # Technical indicators
        ((pl.col(price_col) - pl.col(price_col).rolling_min(7).fill_null(strategy="forward")) / 
         (pl.col(price_col).rolling_max(7).fill_null(strategy="forward") - 
          pl.col(price_col).rolling_min(7).fill_null(strategy="forward"))).alias(f'{price_col}_stochastic'),
        
        # EMA features
        (pl.col(price_col).ewm_mean(alpha=0.1).fill_null(strategy="forward")).alias(f'{price_col}_ema10'),
        (pl.col(price_col).ewm_mean(alpha=0.3).fill_null(strategy="forward")).alias(f'{price_col}_ema3'),
        
        # Price position features
        ((pl.col(price_col) - pl.col(price_col).rolling_min(12).fill_null(strategy="forward")) / 
         (pl.col(price_col).rolling_max(12).fill_null(strategy="forward") - 
          pl.col(price_col).rolling_min(12).fill_null(strategy="forward"))).alias(f'{price_col}_position12'),
        
        # Cyclical features (hour of day, day of week if available)
        # Normalize price features
        ((pl.col(price_col) - pl.col(price_col).mean()) / pl.col(price_col).std()).alias(f'{price_col}_normalized'),
    ])
    
    return df

# Builds an advanced LSTM model with attention mechanism
def build_advanced_lstm_model(input_shape, dropout_rate=0.2):
    if not LSTM_AVAILABLE or tf is None:
        raise RuntimeError("TensorFlow not available for LSTM model creation.")
    
    inputs = Input(shape=input_shape)
    
    # First bidirectional LSTM layer
    lstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))(inputs)
    lstm1 = BatchNormalization()(lstm1)
    
    # Second bidirectional LSTM layer
    lstm2 = Bidirectional(LSTM(64, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    
    # Third LSTM layer
    lstm3 = LSTM(32, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)(lstm2)
    lstm3 = BatchNormalization()(lstm3)
    
    # Global average pooling
    pooled = GlobalAveragePooling1D()(lstm3)
    
    # Dense layers with residual connections
    dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(pooled)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(dropout_rate)(dense1)
    
    dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(dropout_rate)(dense2)
    
    dense3 = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(dense2)
    dense3 = Dropout(dropout_rate)(dense3)
    
    outputs = Dense(1, kernel_regularizer=l2(0.001))(dense3)
    
    # Compile with MAE loss and lower learning rate
    optimizer = Adam(learning_rate=0.0001)  # Lower learning rate
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])  # Use MAE loss directly
    
    return model

# Builds a simpler but effective LSTM model
def build_lstm_model(input_shape, dropout_rate=0.15):  # Reduced dropout
    if not LSTM_AVAILABLE or tf is None:
        raise RuntimeError("TensorFlow not available for LSTM model creation.")
    
    inputs = Input(shape=input_shape)
    
    # Enhanced architecture with proper regularization
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))(inputs)
    x = BatchNormalization()(x)
    
    x = Bidirectional(LSTM(32, return_sequences=False, dropout=dropout_rate, recurrent_dropout=dropout_rate))(x)
    x = BatchNormalization()(x)
    
    # Dense layers with L2 regularization
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.0005))(x)  # Reduced regularization
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(1, kernel_regularizer=l2(0.0005))(x)
    
    # Use MAE loss for direct optimization
    optimizer = Adam(learning_rate=0.0005)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    
    return model

# Predicts future prices using a trained LSTM model
def predict_future_lstm(model, scaler, df, seq_length, periods=24, price_type='buyprice'):
    try:
        # Engineer features for prediction
        df_engineered = engineer_features(df, price_type)
        
        # Get the last sequence
        feature_cols = [
            price_type, f'{price_type}_diff', f'{price_type}_lag1', f'{price_type}_lag2', f'{price_type}_lag3',
            f'{price_type}_ma3', f'{price_type}_ma5', f'{price_type}_ma7', f'{price_type}_ma12',
            f'{price_type}_std3', f'{price_type}_std7', f'{price_type}_momentum3', f'{price_type}_momentum7',
            f'{price_type}_range3', f'{price_type}_return', f'{price_type}_return3', f'{price_type}_stochastic',
            f'{price_type}_ema10', f'{price_type}_ema3', f'{price_type}_position12', f'{price_type}_normalized'
        ]
        
        last_sequence = []
        for col in feature_cols:
            last_vals = df_engineered.select(pl.col(col)).tail(seq_length).to_numpy().astype(np.float32)
            last_sequence.append(last_vals.flatten())
        
        last_sequence = np.column_stack(last_sequence)
        last_sequence_scaled = scaler.transform(last_sequence)
        
        timestamps_np = df.select("timestampdt").to_numpy()
        last_timestamp_np = timestamps_np[-1][0]
        
        if isinstance(last_timestamp_np, np.datetime64):
            last_timestamp = convert_numpy_datetime64_to_python_datetime(last_timestamp_np)
        else:
            last_timestamp = last_timestamp_np
            
        predictions = []
        current_sequence = last_sequence_scaled.copy()
        
        # Store recent values for feature engineering
        recent_prices = list(last_sequence[:, 0][-12:])  # Keep last 12 prices for rolling calculations
        
        # Progress bar for prediction
        pred_pbar = tqdm(range(periods), desc=f"Predicting {price_type}", leave=False)
        
        for i in pred_pbar:
            X_input = current_sequence.reshape((1, seq_length, len(feature_cols)))
            next_price_scaled = model.predict(X_input, verbose=0)[0, 0]
            
            # Create dummy array for inverse transform
            dummy_array = np.zeros((1, len(feature_cols)))
            dummy_array[0, 0] = next_price_scaled
            next_price = scaler.inverse_transform(dummy_array)[0, 0]
            
            next_timestamp = last_timestamp + timedelta(hours=i+1)
            predictions.append({
                "timestamp": next_timestamp,
                "predicted_price": float(next_price)
            })
            
            # Update recent prices
            recent_prices.append(next_price_scaled)
            if len(recent_prices) > 12:
                recent_prices.pop(0)
            
            # Update sequence for next prediction with accurate feature engineering
            new_row = np.zeros(len(feature_cols))
            new_row[0] = next_price_scaled  # price
            
            # Update other features based on prediction
            if len(current_sequence) > 0:
                # Basic features
                new_row[1] = next_price_scaled - current_sequence[-1, 0]  # diff
                new_row[2] = current_sequence[-1, 0]  # lag1
                new_row[3] = current_sequence[-2, 0] if len(current_sequence) > 1 else 0  # lag2
                new_row[4] = current_sequence[-3, 0] if len(current_sequence) > 2 else 0  # lag3
                
                # Moving averages (using recent prices)
                window3 = recent_prices[-3:] if len(recent_prices) >= 3 else recent_prices
                window5 = recent_prices[-5:] if len(recent_prices) >= 5 else recent_prices
                window7 = recent_prices[-7:] if len(recent_prices) >= 7 else recent_prices
                window12 = recent_prices
                
                new_row[5] = np.mean(window3) if window3 else 0  # ma3
                new_row[6] = np.mean(window5) if window5 else 0  # ma5
                new_row[7] = np.mean(window7) if window7 else 0  # ma7
                new_row[8] = np.mean(window12) if window12 else 0  # ma12
                
                # Volatility
                new_row[9] = np.std(window3) if len(window3) > 1 else 0  # std3
                new_row[10] = np.std(window7) if len(window7) > 1 else 0  # std7
                
                # Momentum
                new_row[11] = next_price_scaled - (recent_prices[-4] if len(recent_prices) >= 4 else next_price_scaled)  # momentum3
                new_row[12] = next_price_scaled - (recent_prices[-8] if len(recent_prices) >= 8 else next_price_scaled)  # momentum7
                
                # Range
                new_row[13] = (np.max(window3) - np.min(window3)) if len(window3) > 1 else 0  # range3
                
                # Returns
                new_row[14] = (next_price_scaled / current_sequence[-1, 0] - 1) if current_sequence[-1, 0] != 0 else 0  # return
                new_row[15] = (next_price_scaled / (recent_prices[-4] if len(recent_prices) >= 4 else next_price_scaled) - 1) if (recent_prices[-4] if len(recent_prices) >= 4 else next_price_scaled) != 0 else 0  # return3
                
                # Stochastic
                min7 = np.min(window7) if window7 else next_price_scaled
                max7 = np.max(window7) if window7 else next_price_scaled
                new_row[16] = (next_price_scaled - min7) / (max7 - min7) if (max7 - min7) != 0 else 0  # stochastic
                
                # EMA (simplified calculation)
                new_row[17] = 0.1 * next_price_scaled + 0.9 * current_sequence[-1, 17] if len(current_sequence) > 0 else next_price_scaled  # ema10
                new_row[18] = 0.3 * next_price_scaled + 0.7 * current_sequence[-1, 18] if len(current_sequence) > 0 else next_price_scaled  # ema3
                
                # Position
                min12 = np.min(window12) if window12 else next_price_scaled
                max12 = np.max(window12) if window12 else next_price_scaled
                new_row[19] = (next_price_scaled - min12) / (max12 - min12) if (max12 - min12) != 0 else 0  # position12
                
                # Normalized (simplified)
                mean_prices = np.mean(recent_prices) if recent_prices else next_price_scaled
                std_prices = np.std(recent_prices) if len(recent_prices) > 1 else 1
                new_row[20] = (next_price_scaled - mean_prices) / std_prices if std_prices != 0 else 0  # normalized
            
            current_sequence = np.vstack([current_sequence[1:], new_row])
            
        return predictions
    except Exception as e:
        logging.error(f"Error predicting future with LSTM (internal): {str(e)}")
        logging.debug(traceback.format_exc())
        raise

# Wrapper to process an item with LSTM, returning predictions for BUY AND SELL
def process_item_lstm_primary(df, item_id, periods=24, seq_length=48, epochs=200):  # Reduced epochs
    """
    Wrapper to process item with LSTM, returning predictions for BUY AND SELL.
    Returns:
    (success_flag,
     buy_predictions_list, sell_predictions_list,
     lstm_metrics_dict, message,
     last_buy_price, last_sell_price, last_timestamp)
    lstm_metrics_dict contains in-sample MAE/R2 for BUY price.
    """
    if not LSTM_AVAILABLE or tf is None:
        return False, None, None, None, None, "TensorFlow not available", None, None, None
    
    try:
        item_df = df.filter(pl.col("itemid") == item_id)
        if len(item_df) < seq_length + 150:  # Adjusted minimum requirement
            return False, None, None, None, None, f"Not enough data points ({len(item_df)} < {seq_length + 150})", None, None, None

        # Engineer features for both buy and sell
        item_df_buy = engineer_features(item_df, 'buyprice')
        item_df_sell = engineer_features(item_df, 'sellprice')
        
        # Extract prices and timestamps
        buy_prices = item_df.select("buyprice").to_numpy().astype(np.float32).flatten()
        sell_prices = item_df.select("sellprice").to_numpy().astype(np.float32).flatten()
        timestamps_np = item_df.select("timestampdt").to_numpy()
        
        # More robust outlier handling with winsorization
        def winsorize(data, limits=(0.01, 0.99)):
            lower = np.percentile(data, limits[0] * 100)
            upper = np.percentile(data, limits[1] * 100)
            return np.clip(data, lower, upper)
        
        buy_prices_clean = winsorize(buy_prices)
        sell_prices_clean = winsorize(sell_prices)

        # --- Process BUY Price Model ---
        feature_cols_buy = [
            'buyprice', 'buyprice_diff', 'buyprice_lag1', 'buyprice_lag2', 'buyprice_lag3',
            'buyprice_ma3', 'buyprice_ma5', 'buyprice_ma7', 'buyprice_ma12',
            'buyprice_std3', 'buyprice_std7', 'buyprice_momentum3', 'buyprice_momentum7',
            'buyprice_range3', 'buyprice_return', 'buyprice_return3', 'buyprice_stochastic',
            'buyprice_ema10', 'buyprice_ema3', 'buyprice_position12', 'buyprice_normalized'
        ]
        
        # Prepare data matrix
        buy_data = {}
        for col in feature_cols_buy:
            buy_data[col] = item_df_buy.select(pl.col(col)).to_numpy().astype(np.float32).flatten()
        
        # Use RobustScaler for better outlier handling
        scaler_buy = RobustScaler()
        scaled_features_buy = np.column_stack([buy_data[col] for col in feature_cols_buy])
        scaled_features_buy = scaler_buy.fit_transform(scaled_features_buy)
        
        # Reconstruct scaled data dict
        for i, col in enumerate(feature_cols_buy):
            buy_data[col] = scaled_features_buy[:, i]
        
        logging.info(f"LSTM Primary - Scaled buy prices for {item_id}: median={float(scaler_buy.center_[0])}, scale={float(scaler_buy.scale_[0])}")
        
        X_buy, y_buy = create_sequences(buy_data, seq_length, feature_cols_buy)
        
        if len(X_buy) < 80:  # Adjusted minimum sequences
            return False, None, None, None, None, f"Not enough buy sequences after creation ({len(X_buy)} < 80)", None, None, None

        # Shuffle data before splitting
        indices = np.arange(len(X_buy))
        np.random.shuffle(indices)
        X_buy = X_buy[indices]
        y_buy = y_buy[indices]
        
        split_idx_buy = int(len(X_buy) * 0.85)  # Larger training set
        X_train_buy, X_test_buy = X_buy[:split_idx_buy], X_buy[split_idx_buy:]
        y_train_buy, y_test_buy = y_buy[:split_idx_buy], y_buy[split_idx_buy:]
        
        # Build and train model
        model_buy = build_lstm_model((X_train_buy.shape[1], X_train_buy.shape[2]), dropout_rate=0.15)
        
        # Enhanced callbacks
        early_stopping_buy = EarlyStopping(
            monitor='val_mae', 
            patience=25,  # Reduced patience
            restore_best_weights=True,
            mode='min'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            mode='min',
            verbose=0
        )
        
        model_checkpoint = ModelCheckpoint(
            filepath=f"temp_{item_id}_buy_best.keras",
            monitor='val_mae',
            save_best_only=True,
            mode='min',
            verbose=0
        )
        
        progress_callback = TQDMProgressCallback(epochs=epochs, verbose=1)
        
        print(f"\nTraining BUY model for {item_id}...")
        history_buy = model_buy.fit(
            X_train_buy, y_train_buy, 
            epochs=epochs, 
            batch_size=64,  # Reduced batch size for better gradient updates
            validation_data=(X_test_buy, y_test_buy),
            callbacks=[early_stopping_buy, reduce_lr, model_checkpoint, progress_callback],
            verbose=0
        )

        # Load best model
        model_buy.load_weights(f"temp_{item_id}_buy_best.keras")
        os.remove(f"temp_{item_id}_buy_best.keras")

        # In-sample performance metrics
        y_train_pred_buy = model_buy.predict(X_train_buy, verbose=0).flatten()
        train_mae_buy = mean_absolute_error(y_train_buy, y_train_pred_buy)
        train_r2_buy = r2_score(y_train_buy, y_train_pred_buy)
        final_val_mae_buy = history_buy.history['val_mae'][-1] if 'val_mae' in history_buy.history else np.nan
        
        logging.info(f"LSTM Primary - Buy Model for {item_id}: Train MAE={train_mae_buy:.6f}, Train R2={train_r2_buy:.6f}, Final Val MAE={final_val_mae_buy:.6f}")

        # Save models
        model_dir = "integrated_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path_buy = os.path.join(model_dir, f"{item_id}_lstm_buy.keras")
        model_buy.save(model_path_buy)
        scaler_path_buy = os.path.join(model_dir, f"{item_id}_scaler_lstm_buy.joblib")
        joblib.dump(scaler_buy, scaler_path_buy)

        # Predict future buy prices
        future_buy_predictions = predict_future_lstm(model_buy, scaler_buy, item_df, seq_length, periods=periods, price_type='buyprice')
        
        # --- Process SELL Price Model ---
        feature_cols_sell = [
            'sellprice', 'sellprice_diff', 'sellprice_lag1', 'sellprice_lag2', 'sellprice_lag3',
            'sellprice_ma3', 'sellprice_ma5', 'sellprice_ma7', 'sellprice_ma12',
            'sellprice_std3', 'sellprice_std7', 'sellprice_momentum3', 'sellprice_momentum7',
            'sellprice_range3', 'sellprice_return', 'sellprice_return3', 'sellprice_stochastic',
            'sellprice_ema10', 'sellprice_ema3', 'sellprice_position12', 'sellprice_normalized'
        ]
        
        # Prepare data matrix
        sell_data = {}
        for col in feature_cols_sell:
            sell_data[col] = item_df_sell.select(pl.col(col)).to_numpy().astype(np.float32).flatten()
        
        # Scale features
        scaler_sell = RobustScaler()
        scaled_features_sell = np.column_stack([sell_data[col] for col in feature_cols_sell])
        scaled_features_sell = scaler_sell.fit_transform(scaled_features_sell)
        
        # Reconstruct scaled data dict
        for i, col in enumerate(feature_cols_sell):
            sell_data[col] = scaled_features_sell[:, i]
        
        logging.info(f"LSTM Primary - Scaled sell prices for {item_id}: median={float(scaler_sell.center_[0])}, scale={float(scaler_sell.scale_[0])}")
        
        X_sell, y_sell = create_sequences(sell_data, seq_length, feature_cols_sell)
        
        if len(X_sell) < 80:  # Adjusted minimum sequences
            logging.warning(f"LSTM Primary - Not enough sell sequences for {item_id} ({len(X_sell)} < 80). Proceeding with buy only.")
            future_sell_predictions = [{"timestamp": p["timestamp"], "predicted_price": np.nan} for p in future_buy_predictions]
        else:
            # Shuffle data before splitting
            indices = np.arange(len(X_sell))
            np.random.shuffle(indices)
            X_sell = X_sell[indices]
            y_sell = y_sell[indices]
            
            split_idx_sell = int(len(X_sell) * 0.85)
            X_train_sell, X_test_sell = X_sell[:split_idx_sell], X_sell[split_idx_sell:]
            y_train_sell, y_test_sell = y_sell[:split_idx_sell], y_sell[split_idx_sell:]
            
            model_sell = build_lstm_model((X_train_sell.shape[1], X_train_sell.shape[2]), dropout_rate=0.15)
            
            early_stopping_sell = EarlyStopping(
                monitor='val_mae', 
                patience=25,
                restore_best_weights=True,
                mode='min'
            )
            
            reduce_lr_sell = ReduceLROnPlateau(
                monitor='val_mae',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                mode='min',
                verbose=0
            )
            
            model_checkpoint_sell = ModelCheckpoint(
                filepath=f"temp_{item_id}_sell_best.keras",
                monitor='val_mae',
                save_best_only=True,
                mode='min',
                verbose=0
            )
            
            progress_callback_sell = TQDMProgressCallback(epochs=epochs, verbose=1)
            
            print(f"\nTraining SELL model for {item_id}...")
            history_sell = model_sell.fit(
                X_train_sell, y_train_sell, 
                epochs=epochs, 
                batch_size=64,
                validation_data=(X_test_sell, y_test_sell),
                callbacks=[early_stopping_sell, reduce_lr_sell, model_checkpoint_sell, progress_callback_sell],
                verbose=0
            )
            
            # Load best model
            model_sell.load_weights(f"temp_{item_id}_sell_best.keras")
            os.remove(f"temp_{item_id}_sell_best.keras")
            
            # Save sell model
            model_path_sell = os.path.join(model_dir, f"{item_id}_lstm_sell.keras")
            model_sell.save(model_path_sell)
            scaler_path_sell = os.path.join(model_dir, f"{item_id}_scaler_lstm_sell.joblib")
            joblib.dump(scaler_sell, scaler_path_sell)
            
            future_sell_predictions = predict_future_lstm(model_sell, scaler_sell, item_df, seq_length, periods=periods, price_type='sellprice')

        # Finalize results
        last_buy_price = float(buy_prices[-1])
        last_sell_price = float(sell_prices[-1])
        last_timestamp_np = timestamps_np[-1][0]
        
        if isinstance(last_timestamp_np, np.datetime64):
            last_timestamp = convert_numpy_datetime64_to_python_datetime(last_timestamp_np)
        else:
            last_timestamp = last_timestamp_np

        train_metrics = {
            "mae": float(train_mae_buy), 
            "r2": float(train_r2_buy), 
            "final_val_mae": float(final_val_mae_buy)
        }
        
        return True, future_buy_predictions, future_sell_predictions, train_metrics, "Success", last_buy_price, last_sell_price, last_timestamp

    except Exception as e:
        error_msg = f"LSTM Primary - Error processing item {item_id}: {str(e)}"
        logging.error(error_msg)
        logging.debug(traceback.format_exc())
        return False, None, None, None, None, error_msg, None, None, None

# Combines ARIMA/LSTM predictions using a simple average (50/50 weight) for a single price type
def combine_predictions_simple_average_single(preds_1, preds_2, item_id, price_type="price"):
    """
    Combines two lists of predictions (for the same price type) using a simple average.
    Assumes preds are lists of dicts with 'timestamp' and 'predicted_price'.
    """
    try:
        if not preds_1 or not preds_2 or len(preds_1) != len(preds_2):
            logging.error(f"Cannot combine {price_type} predictions for {item_id}: Mismatch in lists.")
            return None
            
        combined_preds = []
        for p1, p2 in zip(preds_1, preds_2):
            ts_1 = p1["timestamp"]
            ts_2 = p2["timestamp"]
            
            if ts_1 != ts_2:
                logging.error(f"Timestamp mismatch in ensemble for {item_id} ({price_type}): {ts_1} vs {ts_2}")
                return None
                
            # Simple 50/50 average
            combined_price = 0.5 * p1["predicted_price"] + 0.5 * p2["predicted_price"]
            combined_preds.append({
                "timestamp": ts_1,
                "predicted_price": float(combined_price)
            })
            
        logging.info(f"Created simple average ensemble for {item_id} ({price_type}).")
        return combined_preds
    except Exception as e:
        logging.error(f"Error combining {price_type} predictions for {item_id}: {e}")
        logging.debug(traceback.format_exc())
        return None

# Advanced ensemble combining VECM, ARIMA, and LSTM with weighted averaging
def create_weighted_ensemble(vecm_preds, arima_preds, lstm_preds, weights=None):
    """
    Creates a weighted ensemble of three prediction models.
    Default weights: VECM=0.5, ARIMA=0.3, LSTM=0.2 (based on your MAE values)
    """
    if weights is None:
        # Use inverse MAE weighting based on your provided values
        # VECM: 0.116, ARIMA: 0.195, LSTM: 0.463
        total_inverse_mae = (1/0.116) + (1/0.195) + (1/0.463)
        weights = {
            'vecm': (1/0.116) / total_inverse_mae,
            'arima': (1/0.195) / total_inverse_mae,
            'lstm': (1/0.463) / total_inverse_mae
        }
    
    if not vecm_preds or not arima_preds or not lstm_preds:
        return None
    
    if not (len(vecm_preds) == len(arima_preds) == len(lstm_preds)):
        logging.error("Prediction lists have different lengths")
        return None
    
    ensemble_preds = []
    for v, a, l in zip(vecm_preds, arima_preds, lstm_preds):
        # Check timestamp alignment
        if not (v["timestamp"] == a["timestamp"] == l["timestamp"]):
            logging.error("Timestamp mismatch in ensemble")
            return None
        
        # Weighted average
        ensemble_price = (
            weights['vecm'] * v["predicted_price"] +
            weights['arima'] * a["predicted_price"] +
            weights['lstm'] * l["predicted_price"]
        )
        
        ensemble_preds.append({
            "timestamp": v["timestamp"],
            "predicted_price": float(ensemble_price)
        })
    
    return ensemble_preds