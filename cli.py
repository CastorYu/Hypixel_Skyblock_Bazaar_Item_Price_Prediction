# --- Standard Library Imports ---
# Import necessary libraries for data manipulation, numerical operations,
# file handling, logging, and command-line argument parsing.
import polars as pl         # Efficient DataFrame library for data processing
import numpy as np          # Numerical computing library
import json                 # Library for handling JSON data
import os                   # Library for interacting with the operating system
import logging              # Library for logging events and messages
from datetime import datetime, timedelta, timezone # Date and time handling
import warnings             # Library to manage warning messages
import argparse             # Library for parsing command-line arguments
from sklearn.metrics import mean_absolute_error, r2_score # Metrics for model evaluation

# --- ARIMA Model Imports ---
# Import libraries required for ARIMA (AutoRegressive Integrated Moving Average) modeling.
from statsmodels.tsa.arima.model import ARIMA # ARIMA model implementation
from statsmodels.tsa.stattools import adfuller # Augmented Dickey-Fuller test for stationarity

# --- LSTM Model Imports (Optional) ---
# Import libraries for LSTM (Long Short-Term Memory) neural networks.
# Handle potential import errors gracefully if TensorFlow is not installed.
try:
    import tensorflow as tf                 # Deep learning framework
    from sklearn.preprocessing import StandardScaler # Scaler for normalizing data
    LSTM_AVAILABLE = True                   # Flag indicating LSTM availability
    tf.get_logger().setLevel('ERROR')       # Reduce TensorFlow logging verbosity
except ImportError:
    LSTM_AVAILABLE = False                  # Flag indicating LSTM unavailability
    logging.warning("TensorFlow not found. LSTM models will be disabled.")

# --- VECM Model Imports (Optional) ---
# Import libraries for VECM (Vector Error Correction Model).
# Handle potential import errors gracefully if statsmodels VECM is not available.
try:
    from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank # VECM implementation and cointegration rank selection
    VECM_AVAILABLE = True                   # Flag indicating VECM availability
except ImportError:
    VECM_AVAILABLE = False                  # Flag indicating VECM unavailability
    logging.warning("statsmodels VECM not found or version too old. VECM models will be disabled.")

# --- Additional Imports ---
# Import other necessary libraries.
import traceback            # Library for printing detailed error information
import joblib               # Library for saving and loading Python objects
import matplotlib.pyplot as plt # Library for creating plots
import pandas as pd         # Library for data manipulation (used as a fallback for datetime conversion)

# --- Logging Configuration ---
# Configure logging to write messages to a file and the console with UTF-8 encoding.
# Keep the original logging setup from Pasted_Text_1754634156063.txt
logging.basicConfig(
    level=logging.INFO, # Default logging level (INFO and above)
    format='%(asctime)s - %(levelname)s - %(message)s', # Log message format
    handlers=[
        logging.FileHandler('cli.log', encoding='utf-8'), # Log file with UTF-8 encoding
        logging.StreamHandler() # Console output handler (always added)
    ]
)

# --- Windows Console Encoding Fix ---
# Set console encoding to UTF-8 on Windows to handle special characters correctly.
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# --- Helper Function: Datetime Conversion ---
# Converts numpy.datetime64 objects to timezone-aware Python datetime objects (UTC).
# Uses Polars for conversion, with a fallback to Pandas if Polars fails.
def convert_numpy_datetime64_to_python_datetime(np_dt64):
    """
    Converts a numpy.datetime64 object to a timezone-aware Python datetime object (UTC).
    Uses Polars for robust conversion, with a fallback to Pandas/NumPy if Polars fails.
    """
    if not isinstance(np_dt64, np.datetime64):
        raise TypeError(f"Input must be a numpy.datetime64 object, got {type(np_dt64)}")
    try:
        # --- Attempt 1: Using Polars (Standard Method) ---
        # Create a Polars Series, cast to Datetime, convert to list
        ts_pl_series = pl.Series([np_dt64]).cast(pl.Datetime)
        # Polars typically returns timezone-naive datetime, so we add UTC tzinfo
        converted_datetime = ts_pl_series.to_list()[0] # Get the first (and only) element
        # Ensure the result is timezone-aware in UTC
        if converted_datetime.tzinfo is None:
            converted_datetime = converted_datetime.replace(tzinfo=timezone.utc)
        return converted_datetime
    except Exception as e_pl:
        logging.warning(f"Polars conversion failed for {np_dt64}: {e_pl}. Trying Pandas fallback.")
        try:
            # --- Attempt 2: Fallback using Pandas/NumPy ---
            # Convert numpy.datetime64 to Python datetime using Pandas
            # pd.to_datetime handles various numpy datetime64 dtypes well
            converted_datetime = pd.to_datetime(np_dt64).to_pydatetime()
            # Ensure it's timezone-aware if needed (Pandas often attaches tzinfo, but be safe)
            if converted_datetime.tzinfo is None:
                converted_datetime = converted_datetime.replace(tzinfo=timezone.utc)
            return converted_datetime
        except Exception as e_pd:
            # --- Final Error ---
            error_msg = f"Error converting datetime64 {np_dt64}: Polars error: {e_pl}, Pandas error: {e_pd}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e_pd # Chain the last exception

# --- Data Loading Function ---
# Loads data from an NDJSON file with robust datetime parsing using Polars, falling back to Pandas.
def load_data(file_path):
    """Loads data with robust datetime parsing using Polars, falling back to Pandas."""
    try:
        # --- Attempt 1: Load and parse with Polars ---
        try:
            df = pl.read_ndjson(file_path) # Read NDJSON file into a Polars DataFrame
            # Rename columns to match expected schema
            df = df.rename({
                "item_id": "itemid",
                "buy_price": "buyprice",
                "sell_price": "sellprice",
                "buy_volume": "buyvolume",
                "sell_volume": "sellvolume",
                "buy_moving_week": "buymovingweek",
                "sell_moving_week": "sellmovingweek",
                "buy_orders": "buyorders",
                "sell_orders": "sellorders"
            })
            # Parse timestamp using Polars
            df = df.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("timestampdt")
            )
            logging.info(f"Loaded {len(df)} records using Polars parsing.")
        except Exception as e_pl_parse:
            logging.warning(f"Polars parsing failed: {e_pl_parse}. Trying Pandas fallback for loading.")
            # --- Attempt 2: Fallback using Pandas for parsing ---
            try:
                # Load with Pandas
                pdf = pd.read_json(file_path, lines=True) # Read NDJSON file into a Pandas DataFrame
                # Rename columns to match expected schema
                pdf.rename(columns={
                    "item_id": "itemid",
                    "buy_price": "buyprice",
                    "sell_price": "sellprice",
                    "buy_volume": "buyvolume",
                    "sell_volume": "sellvolume",
                    "buy_moving_week": "buymovingweek",
                    "sell_moving_week": "sellmovingweek",
                    "buy_orders": "buyorders",
                    "sell_orders": "sellorders"
                }, inplace=True)
                # Parse timestamp using Pandas
                pdf['timestampdt'] = pd.to_datetime(pdf['timestamp'], format="%Y-%m-%d %H:%M:%S")
                # Convert to Polars DataFrame
                df = pl.from_pandas(pdf)
                logging.info(f"Loaded {len(df)} records using Pandas fallback for parsing.")
            except Exception as e_pd_load:
                 error_msg = f"Error loading data with Polars ({e_pl_parse}) and Pandas ({e_pd_load})"
                 logging.error(error_msg)
                 raise RuntimeError(error_msg) from e_pd_load
        # Ensure the timestampdt column is strictly Datetime type and sort
        df = df.with_columns(pl.col("timestampdt").cast(pl.Datetime)).sort("timestampdt")
        return df
    except Exception as e:
        logging.error(f"Error loading  {str(e)}")
        raise

# --- Constant Series Detection ---
# Checks if a price series is effectively constant.
def is_price_constant(prices, rel_tol=1e-3):
    """
    Checks if the price series is effectively constant.
    Args:
        prices (np.ndarray): 1D array of prices.
        rel_tol (float): Relative tolerance for standard deviation compared to mean.
                         Default 1e-3 means std < 0.1% of mean is considered constant.
    Returns:
        bool: True if constant/near-constant, False otherwise.
    """
    if len(prices) < 2:
        return True # Not enough data, treat as constant
    mean_price = np.mean(prices)
    if mean_price == 0:
        # Avoid division by zero. If all prices are zero, it's constant.
        return np.all(prices == 0)
    std_price = np.std(prices)
    #logging.debug(f"Checking constancy for item: mean={mean_price:.6f}, std={std_price:.6f}, ratio={std_price / mean_price:.6f}")
    return (std_price / abs(mean_price)) < rel_tol

# Generates future predictions for a constant price series.
def predict_future_constant(last_price, last_timestamp, periods=24):
    """Generates future predictions for a constant price series."""
    predictions = []
    for i in range(periods):
        next_timestamp = last_timestamp + timedelta(hours=i+1)
        predictions.append({
            "timestamp": next_timestamp,
            "predicted_price": float(last_price)
        })
    return predictions

# Saves results and plots for a constant price series.
def save_and_plot_constant(item_id, last_price, last_timestamp, future_predictions, item_df_for_plotting):
    """Saves results and plots for a constant price series."""
    try:
        pred_dir = "integrated_predictions"
        os.makedirs(pred_dir, exist_ok=True)
        pred_path = os.path.join(pred_dir, f"{item_id}.json")
        predictions_data = {
            "item_id": item_id,
            "current_price": float(last_price),
            "current_timestamp": last_timestamp.isoformat(),
            "best_model_type": "Constant (Naive)",
            "models": {
                "Constant": {
                    "method": "Naive (Last Price Repeated)",
                    "performance": {"note": "Price is constant, no training metrics"},
                    "future_predictions": [
                        {"timestamp": p["timestamp"].isoformat(), "predicted_price": float(p["predicted_price"])}
                        for p in future_predictions
                    ]
                }
            }
        }
        with open(pred_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        logging.info(f"Saved constant model results for {item_id} to {pred_path}")
        # --- Plotting ---
        plt.figure(figsize=(12, 6))
        plot_points = min(200, len(item_df_for_plotting))
        timestamps_hist_np = item_df_for_plotting.select("timestampdt").tail(plot_points).to_numpy().flatten()
        if len(timestamps_hist_np) > 0 and isinstance(timestamps_hist_np[0], np.datetime64):
            timestamps_hist = [
                convert_numpy_datetime64_to_python_datetime(ts)
                for ts in timestamps_hist_np
            ]
        else:
             timestamps_hist = timestamps_hist_np
        prices_hist = item_df_for_plotting.select("buyprice").tail(plot_points).to_numpy().flatten()
        plt.plot(timestamps_hist, prices_hist, label='Historical Price', color='blue')
        future_timestamps = [p["timestamp"] for p in future_predictions]
        future_prices = [p["predicted_price"] for p in future_predictions]
        if len(timestamps_hist) > 0 and len(future_timestamps) > 0:
            conn_hist_ts = timestamps_hist[-1]
            conn_pred_ts = future_timestamps[0]
            conn_hist_price = prices_hist[-1]
            conn_pred_price = future_prices[0]
            plt.plot([conn_hist_ts, conn_pred_ts], [conn_hist_price, conn_pred_price],
                     color='red', linestyle='--', alpha=0.7)
        plt.plot(future_timestamps, future_prices, linestyle='--', marker='o',
                 label='Predicted Price (Constant)', color='red')
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.title(f"Price Prediction for {item_id} (Constant Model)")
        plt.legend()
        plt.grid(True)
        plot_dir = "integrated_plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{item_id}.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved constant model plot for {item_id} to {plot_path}")
        # --- End Plotting ---
    except Exception as e:
        logging.error(f"Error saving/plotting constant model results for {item_id}: {e}")
# --- End Constant Series Handling ---

# --- VECM Helper Functions ---
# Wrapper to process an item with VECM using multiple features.
def process_item_vecm(df, item_id, periods=24, max_lag=3, max_coint_rank=2, train_split_ratio=0.8): # Added periods
    """
    Wrapper to process item with VECM using multiple features.
    Returns: (success_flag, predictions_list, metrics_dict, message, last_price, last_timestamp)
    """
    if not VECM_AVAILABLE:
        return False, None, None, "VECM not available", None, None
    try:
        item_df = df.filter(pl.col("itemid") == item_id) # Filter data for the specific item
        min_data_points = 50 # Minimum required data points
        if len(item_df) < min_data_points:
            return False, None, None, f"Not enough data points ({len(item_df)} < {min_data_points})", None, None
        # --- Select relevant features for VECM ---
        # Using price and volume data for the item
        feature_columns = ['buyprice', 'sellprice', 'buyvolume', 'sellvolume']
        # Check if all features exist and have variance
        available_features = []
        for col in feature_columns:
            if col in item_df.columns:
                col_data = item_df.select(col).to_numpy().astype(np.float64).flatten()
                if np.var(col_data) > 1e-10: # Check for non-zero variance
                     available_features.append(col)
                else:
                     logging.info(f"VECM for {item_id}: Feature {col} has zero variance, skipping.")
            else:
                 logging.info(f"VECM for {item_id}: Feature {col} not found, skipping.")
        if len(available_features) < 2:
            return False, None, None, f"Not enough valid features with variance ({len(available_features)} < 2)", None, None
        data_df = item_df.select(available_features).to_pandas() # Convert selected features to Pandas DataFrame
        # --- End Feature Selection ---
        # --- Check for stationarity/cointegration potential ---
        # A quick check: if all series are constant, VECM isn't suitable
        constant_series_count = 0
        for col in data_df.columns:
            if is_price_constant(data_df[col].values, rel_tol=1e-4): # Stricter constancy check
                 constant_series_count += 1
        if constant_series_count == len(data_df.columns):
             return False, None, None, "All VECM features are constant", None, None
        # --- End Stationarity Check ---
        # --- Train/Test Split (for in-sample evaluation) ---
        split_idx = int(len(data_df) * train_split_ratio)
        train_data = data_df.iloc[:split_idx]
        test_data = data_df.iloc[split_idx:] # Used for in-sample evaluation, not forecasting
        # --------------------------------------------------
        # --- Determine Cointegration Rank ---
        # Use the training data for the Johansen test
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning) # Ignore convergence warnings
                coint_rank_result = select_coint_rank(train_data, det_order=1, k_ar_diff=max_lag, method='trace', signif=0.05)
            coint_rank = coint_rank_result.rank if coint_rank_result.rank <= max_coint_rank else max_coint_rank
            logging.info(f"VECM for {item_id}: Selected cointegration rank: {coint_rank} (trace test)")
        except Exception as e:
            logging.warning(f"VECM for {item_id}: Could not determine coint rank using trace (error: {e}). Trying rank=1.")
            coint_rank = 1 # Default fallback
        # --- End Coint Rank ---
        # --- Fit VECM Model ---
        # Fit on the full dataset for best forecast model
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning) # Ignore convergence warnings
                vecm_model = VECM(data_df, k_ar_diff=max_lag, coint_rank=coint_rank, deterministic='co')
                vecm_fitted = vecm_model.fit()
        except Exception as e:
             return False, None, None, f"Failed to fit VECM model: {e}", None, None
        logging.info(f"VECM for {item_id}: Fitted model with k_ar_diff={max_lag}, coint_rank={coint_rank}")
        # --- End Model Fit ---
        # --- In-Sample Evaluation (on training set) ---
        # Get in-sample fitted values for the first variable (buyprice)
        buyprice_col_index = list(data_df.columns).index('buyprice')
        insample_fitted = vecm_fitted.fittedvalues[:, buyprice_col_index] # 2D array, select buyprice column
        insample_actual = train_data['buyprice'].values # 1D array
        # Align lengths (fittedvalues excludes first k_ar_diff points)
        if len(insample_fitted) != len(insample_actual):
            # Trim actual to match fitted length
            insample_actual_aligned = insample_actual[-len(insample_fitted):]
        else:
            insample_actual_aligned = insample_actual
        if len(insample_actual_aligned) > 0 and len(insample_fitted) > 0:
            insample_mae = mean_absolute_error(insample_actual_aligned, insample_fitted)
            # Handle potential R2 issues if actual has zero variance in aligned segment
            if np.var(insample_actual_aligned) > 1e-10:
                insample_r2 = r2_score(insample_actual_aligned, insample_fitted)
            else:
                insample_r2 = np.nan
        else:
            insample_mae = np.nan
            insample_r2 = np.nan
        # --- End In-Sample Eval ---
        logging.info(f"VECM for {item_id}: In-sample MAE (buyprice)={insample_mae:.4f}, R2={insample_r2:.4f}")
        # --- Save Model ---
        model_dir = "integrated_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{item_id}_vecm.pkl")
        joblib.dump(vecm_fitted, model_path)
        # --- End Save ---
        # --- Future Predictions ---
        # Use the passed 'periods' argument
        future_periods = periods
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning) # Ignore forecast warnings
                forecast_result = vecm_fitted.predict(steps=future_periods)
            # Extract 'buyprice' forecasts (first column assumed to be 'buyprice')
            future_buyprice_forecasts = forecast_result[:, buyprice_col_index]
        except Exception as e:
            return False, None, None, f"Failed to forecast with VECM: {e}", None, None
        # --- End Forecasting ---
        # --- Prepare Output ---
        timestamps_np = item_df.select("timestampdt").to_numpy().flatten()
        last_timestamp_np = timestamps_np[-1]
        if isinstance(last_timestamp_np, np.datetime64):
            # Use the fixed conversion function
            last_timestamp = convert_numpy_datetime64_to_python_datetime(last_timestamp_np)
        else:
            last_timestamp = last_timestamp_np
        future_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(future_periods)]
        last_price = float(item_df.select("buyprice").to_numpy().astype(np.float32).flatten()[-1])
        vecm_future_predictions = [
            {"timestamp": ts, "predicted_price": float(pred)}
            for ts, pred in zip(future_timestamps, future_buyprice_forecasts)
        ]
        metrics = {
            "aic": float(vecm_fitted.aic), # AIC of the fitted model
            "insample_mae": float(insample_mae),
            "insample_r2": float(insample_r2) if not np.isnan(insample_r2) else None,
            "coint_rank": int(coint_rank),
            "k_ar_diff": int(max_lag)
        }
        # --- End Prepare Output ---
        return True, vecm_future_predictions, metrics, "Success", last_price, last_timestamp
    except Exception as e:
        error_msg = f"VECM Primary - Error processing item {item_id}: {str(e)}"
        logging.error(error_msg)
        logging.debug(traceback.format_exc())
        return False, None, None, error_msg, None, None
# --- End VECM Helper Functions ---

# --- LSTM Helper Functions ---
# Creates sequences for LSTM training.
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Builds an LSTM model using Keras.
def build_lstm_model(input_shape):
    if not LSTM_AVAILABLE:
        raise RuntimeError("TensorFlow not available for LSTM model creation.")
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(128, activation='relu', return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(64, activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Predicts future prices using a trained LSTM model.
def predict_future_lstm(model, scaler, df, seq_length, periods=24): # Added periods
    try:
        last_prices = df.select("buyprice").tail(seq_length).to_numpy().astype(np.float32)
        last_prices_scaled = scaler.transform(last_prices)
        timestamps_np = df.select("timestampdt").to_numpy()
        last_timestamp_np = timestamps_np[-1][0] # Accessing [0] because to_numpy() returns [[ts]]
        if isinstance(last_timestamp_np, np.datetime64):
            # Use the fixed conversion function
            last_timestamp = convert_numpy_datetime64_to_python_datetime(last_timestamp_np)
        else:
            last_timestamp = last_timestamp_np
        predictions = []
        current_sequence = last_prices_scaled.flatten()
        # Use the passed 'periods' argument
        for i in range(periods):
            X_input = current_sequence.reshape((1, seq_length, 1))
            next_price_scaled = model.predict(X_input, verbose=0)[0, 0]
            next_price = scaler.inverse_transform([[next_price_scaled]])[0, 0]
            next_timestamp = last_timestamp + timedelta(hours=i+1)
            predictions.append({
                "timestamp": next_timestamp,
                "predicted_price": float(next_price)
            })
            current_sequence = np.append(current_sequence[1:], next_price_scaled)
        return predictions
    except Exception as e:
        logging.error(f"Error predicting future with LSTM: {str(e)}")
        raise

# Plots historical data and predictions from ARIMA, LSTM, Ensemble, and VECM.
def plot_predictions_combined(df, arima_preds, lstm_preds, ensemble_preds, vecm_preds, item_id):
    """Plots historical data and predictions from ARIMA, LSTM, Ensemble, and VECM."""
    try:
        plt.figure(figsize=(12, 6))
        plot_points = min(200, len(df))
        timestamps_hist_np = df.select("timestampdt").tail(plot_points).to_numpy().flatten()
        if len(timestamps_hist_np) > 0 and isinstance(timestamps_hist_np[0], np.datetime64):
            # Use the fixed conversion function
            timestamps_hist = [
                convert_numpy_datetime64_to_python_datetime(ts)
                for ts in timestamps_hist_np
            ]
        else:
             timestamps_hist = timestamps_hist_np
        prices_hist = df.select("buyprice").tail(plot_points).to_numpy().flatten()
        plt.plot(timestamps_hist, prices_hist, label='Historical Price', color='blue')
        def plot_pred_lines(preds, label, color, linestyle='--'):
            if not preds:
                return
            pred_timestamps = [p["timestamp"] for p in preds]
            pred_prices = [p["predicted_price"] for p in preds]
            if len(timestamps_hist) > 0 and len(pred_timestamps) > 0:
                conn_hist_ts = timestamps_hist[-1]
                conn_pred_ts = pred_timestamps[0]
                conn_hist_price = prices_hist[-1]
                conn_pred_price = pred_prices[0]
                plt.plot([conn_hist_ts, conn_pred_ts], [conn_hist_price, conn_pred_price],
                         color=color, linestyle=linestyle, alpha=0.5)
            plt.plot(pred_timestamps, pred_prices, linestyle=linestyle, marker='o',
                     label=label, color=color)
        plot_pred_lines(arima_preds, 'ARIMA Pred', 'orange')
        plot_pred_lines(lstm_preds, 'LSTM Pred', 'green')
        plot_pred_lines(ensemble_preds, 'Ensemble Pred (Avg)', 'red')
        plot_pred_lines(vecm_preds, 'VECM Pred', 'purple')
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.title(f"Combined Price Prediction for {item_id}")
        plt.legend()
        plt.grid(True)
        plot_dir = "integrated_plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{item_id}.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved integrated plot for {item_id} to {plot_path}")
    except Exception as e:
        logging.error(f"Error plotting combined predictions for {item_id}: {str(e)}")

# Wrapper to process an item with LSTM, returning predictions and metrics (train/val only).
def process_item_lstm_primary(df, item_id, periods=24, seq_length=24, epochs=50): # Added periods
    """
    Wrapper to process item with LSTM, returning predictions and metrics (train/val only).
    Returns: (success_flag, predictions_list, train_metrics_dict, message, last_price, last_timestamp)
    """
    if not LSTM_AVAILABLE:
        return False, None, None, "TensorFlow not available", None, None
    try:
        item_df = df.filter(pl.col("itemid") == item_id) # Filter data for the specific item
        if len(item_df) < seq_length + 20: # Check for sufficient data
            return False, None, None, f"Not enough data points ({len(item_df)} < {seq_length + 20})", None, None
        prices = item_df.select("buyprice").to_numpy().astype(np.float32) # Extract buy prices
        timestamps_np = item_df.select("timestampdt").to_numpy()
        scaler = StandardScaler() # Initialize scaler
        scaled_prices = scaler.fit_transform(prices) # Scale prices
        logging.info(f"LSTM Primary - Scaled prices for {item_id}: mean={float(scaler.mean_[0])}, std={float(scaler.scale_[0])}")
        X, y = create_sequences(scaled_prices, seq_length) # Create sequences for training
        if len(X) < 20: # Check for sufficient sequences
             return False, None, None, f"Not enough sequences after creation ({len(X)} < 20)", None, None
        split_idx = int(len(X) * 0.8) # Split data for training and testing
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        model = build_lstm_model((X_train.shape[1], X_train.shape[2])) # Build LSTM model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # Early stopping callback
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping],
                            verbose=0) # Train model
        # --- Use In-Sample Performance for Metrics ---
        y_train_pred_scaled = model.predict(X_train, verbose=0) # Predict on training data
        y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten() # Inverse transform actual values
        y_train_pred_inv = scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten() # Inverse transform predicted values
        train_mae = mean_absolute_error(y_train_inv, y_train_pred_inv) # Calculate MAE
        train_r2 = r2_score(y_train_inv, y_train_pred_inv) # Calculate R2
        final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else np.nan # Get final validation loss
        # ---------------------------------------------
        logging.info(f"LSTM Primary - Model for {item_id}: Train MAE={train_mae:.4f}, Train R2={train_r2:.4f}, Final Val Loss={final_val_loss:.6f}")
        model_dir = "integrated_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{item_id}_lstm.keras") # Save model
        model.save(model_path)
        scaler_path = os.path.join(model_dir, f"{item_id}_scaler_lstm.joblib") # Save scaler
        joblib.dump(scaler, scaler_path)
        # Use the passed 'periods' argument
        future_predictions = predict_future_lstm(model, scaler, item_df, seq_length, periods=periods) # Predict future prices
        last_actual_price = float(prices[-1, 0]) # Get last actual price
        last_actual_timestamp_np = timestamps_np[-1][0] # Get last actual timestamp
        if isinstance(last_actual_timestamp_np, np.datetime64):
            # Use the fixed conversion function
            last_actual_timestamp = convert_numpy_datetime64_to_python_datetime(last_actual_timestamp_np)
        else:
            last_actual_timestamp = last_actual_timestamp_np
        train_metrics = {"mae": float(train_mae), "r2": float(train_r2), "final_val_loss": float(final_val_loss)} # Prepare training metrics
        return True, future_predictions, train_metrics, "Success", last_actual_price, last_actual_timestamp
    except Exception as e:
        error_msg = f"LSTM Primary - Error processing item {item_id}: {str(e)}"
        logging.error(error_msg)
        logging.debug(traceback.format_exc())
        return False, None, None, error_msg, None, None
# --- End LSTM Helper Functions ---

# Finds the best ARIMA order using AIC.
def find_best_order(series, max_p=5, max_d=2, max_q=5):
    """Find the best ARIMA order using AIC."""
    if len(np.unique(series)) == 1:
        logging.info("Series is constant. Returning order (0, 0, 0).")
        return (0, 0, 0), None
    best_aic = np.inf
    best_order = None
    best_model = None
    warnings.filterwarnings("ignore") # Ignore convergence warnings during search
    d = 0
    adf_result = adfuller(series)
    if adf_result[1] <= 0.05:
        d = 0 # Series is already stationary
    else:
        for i in range(1, max_d + 1):
            if len(series) <= i:
                 break
            diff_series = np.diff(series, n=i)
            if len(diff_series) == 0:
                break
            adf_result = adfuller(diff_series)
            if adf_result[1] <= 0.05:
                d = i
                break
        else:
            d = max_d # If still not stationary, use max_d
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                order = (p, d, q)
                model = ARIMA(series, order=order)
                fitted_model = model.fit()
                aic = fitted_model.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                    best_model = fitted_model
            except Exception:
                continue # Skip invalid combinations
    warnings.filterwarnings("default")
    return best_order, best_model

# Wrapper to process an item with ARIMA, returning predictions and AIC.
def process_item_arima_primary(df, item_id, periods=24, train_split_ratio=0.8): # Added periods
    """
    Wrapper to process item with ARIMA, returning predictions and AIC.
    Returns: (success_flag, predictions_list, arima_info_dict, message, last_price, last_timestamp)
    arima_info_dict contains 'aic' and potentially in-sample metrics.
    """
    try:
        item_df = df.filter(pl.col("itemid") == item_id) # Filter data for the specific item
        min_data_points = 50 # Minimum required data points
        if len(item_df) < min_data_points:
            return False, None, None, f"Not enough data points ({len(item_df)} < {min_data_points})", None, None
        prices = item_df.select("buyprice").to_numpy().astype(np.float32).flatten() # Extract buy prices
        timestamps_np = item_df.select("timestampdt").to_numpy().flatten()
        if len(np.unique(prices)) == 1: # Check for constant price series
             return False, None, None, "Constant price series", None, None
        # --- Use In-Sample Performance for Metrics ---
        # Find best order on full dataset for final model and AIC
        best_order, fitted_model = find_best_order(prices) # Use full data for AIC/model selection
        if best_order is None or fitted_model is None:
            return False, None, None, "No suitable ARIMA model", None, None
        logging.info(f"ARIMA Primary - Best order for {item_id}: {best_order}")
        final_aic = fitted_model.aic # Get AIC of the fitted model
        # Get in-sample fit statistics
        insample_mae = mean_absolute_error(prices, fitted_model.fittedvalues) # Calculate in-sample MAE
        insample_r2 = r2_score(prices, fitted_model.fittedvalues) # Calculate in-sample R2
        # ---------------------------------------------
        logging.info(f"ARIMA Primary - Model for {item_id}: AIC={final_aic:.4f}, In-sample MAE={insample_mae:.4f}, In-sample R2={insample_r2:.4f}")
        model_dir = "integrated_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{item_id}_arima.pkl") # Save model
        joblib.dump(fitted_model, model_path)
        # --- Future Predictions for ARIMA ---
        # Use the passed 'periods' argument
        future_periods = periods
        try:
            future_forecast_res = fitted_model.get_forecast(steps=future_periods) # Forecast future prices
            future_predictions_mean = future_forecast_res.predicted_mean
        except Exception as e:
            return False, None, None, f"Future forecast error: {e}", None, None
        last_timestamp_np = timestamps_np[-1] # Get last timestamp
        if isinstance(last_timestamp_np, np.datetime64):
            # Use the fixed conversion function
            last_timestamp = convert_numpy_datetime64_to_python_datetime(last_timestamp_np)
        else:
            last_timestamp = last_timestamp_np
        future_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(future_periods)] # Generate future timestamps
        arima_future_predictions = [
            {"timestamp": ts, "predicted_price": float(pred)}
            for ts, pred in zip(future_timestamps, future_predictions_mean)
        ]
        last_actual_price = float(prices[-1]) # Get last actual price
        last_actual_timestamp = last_timestamp # Get last actual timestamp
        # Return AIC and in-sample metrics
        arima_info = {"aic": float(final_aic), "mae": float(insample_mae), "r2": float(insample_r2)} # Prepare ARIMA info
        return True, arima_future_predictions, arima_info, "Success", last_actual_price, last_actual_timestamp
    except Exception as e:
        error_msg = f"ARIMA Primary - Error processing item {item_id}: {str(e)}"
        logging.error(error_msg)
        logging.debug(traceback.format_exc())
        return False, None, None, error_msg, None, None

# Combines ARIMA and LSTM predictions using a simple average (50/50 weight).
def combine_predictions_simple_average(preds_arima, preds_lstm, item_id):
    """
    Combines ARIMA and LSTM predictions using a simple average (50/50 weight).
    Assumes preds_arima and preds_lstm are lists of dicts with 'timestamp' and 'predicted_price'.
    """
    try:
        if not preds_arima or not preds_lstm or len(preds_arima) != len(preds_lstm):
            logging.error(f"Cannot combine predictions for {item_id}: Mismatch in lists.")
            return None
        combined_preds = []
        for pa, pl in zip(preds_arima, preds_lstm):
            ts_arima = pa["timestamp"]
            ts_lstm = pl["timestamp"]
            if ts_arima != ts_lstm:
                logging.error(f"Timestamp mismatch in ensemble for {item_id}: {ts_arima} vs {ts_lstm}")
                return None
            # Simple 50/50 average
            combined_price = 0.5 * pa["predicted_price"] + 0.5 * pl["predicted_price"]
            combined_preds.append({
                "timestamp": ts_arima,
                "predicted_price": float(combined_price)
            })
        logging.info(f"Created simple average ensemble for {item_id}.")
        return combined_preds
    except Exception as e:
        logging.error(f"Error combining predictions for {item_id}: {e}")
        return None

# --- Main Processing Logic (Modified for CLI, but full functionality) ---
# Processes a single item, saving results, models, and plots like the original.
def process_single_item_full(df, item_id, periods=24): # Added periods
    """Processes a single item, saving results, models, and plots like the original."""
    logging.info(f"Processing item: {item_id} for {periods} periods ahead")
    item_df = df.filter(pl.col("itemid") == item_id) # Filter data for the specific item
    if len(item_df) == 0:
        logging.error(f"Item ID '{item_id}' not found in the data.")
        return
    # --- 1. Check for Constant Price ---
    prices_np = item_df.select("buyprice").to_numpy().astype(np.float32).flatten()
    if is_price_constant(prices_np): # Check if price is constant
        logging.info(f"Item {item_id} has constant/near-constant price. Using Naive model.")
        timestamps_np = item_df.select("timestampdt").to_numpy().flatten()
        last_price = float(prices_np[-1]) # Get last price
        last_timestamp_np = timestamps_np[-1] # Get last timestamp
        if isinstance(last_timestamp_np, np.datetime64):
            # Use the fixed conversion function
            last_timestamp = convert_numpy_datetime64_to_python_datetime(last_timestamp_np)
        else:
            last_timestamp = last_timestamp_np
        # Pass the 'periods' argument
        const_preds = predict_future_constant(last_price, last_timestamp, periods=periods) # Predict future prices for constant series
        save_and_plot_constant(item_id, last_price, last_timestamp, const_preds, item_df) # Save and plot results
        logging.info(f"Finished processing item: {item_id} (Constant)")
        return # Move to next item
    # --- End Constant Check ---
    # --- 2. If Not Constant, Try VECM First ---
    logging.info(f"Item {item_id} has varying price. Attempting VECM.")
    # Pass the 'periods' argument
    vecm_success, vecm_preds, vecm_metrics, vecm_msg, vecm_last_price, vecm_last_ts = \
        process_item_vecm(df, item_id, periods=periods, max_lag=2, max_coint_rank=2, train_split_ratio=0.8) # Process item with VECM
    if vecm_success:
        logging.info(f"VECM successful for {item_id}. Using VECM predictions.")
        # --- Save VECM Results ---
        try:
            pred_dir = "integrated_predictions"
            os.makedirs(pred_dir, exist_ok=True)
            pred_path = os.path.join(pred_dir, f"{item_id}.json")
            predictions_data = {
                "item_id": item_id,
                "current_price": vecm_last_price,
                "current_timestamp": vecm_last_ts.isoformat(),
                "best_model_type": "VECM",
                "models": {
                    "VECM": {
                        "info": vecm_metrics,
                        "future_predictions": [
                            {"timestamp": p["timestamp"].isoformat(), "predicted_price": float(p["predicted_price"])}
                            for p in vecm_preds
                        ]
                    }
                }
            }
            with open(pred_path, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            logging.info(f"Saved VECM results for {item_id} to {pred_path}")
            # --- Plot VECM Only ---
            plt.figure(figsize=(12, 6))
            plot_points = min(200, len(item_df))
            timestamps_hist_np = item_df.select("timestampdt").tail(plot_points).to_numpy().flatten()
            if len(timestamps_hist_np) > 0 and isinstance(timestamps_hist_np[0], np.datetime64):
                # Use the fixed conversion function
                timestamps_hist = [
                    convert_numpy_datetime64_to_python_datetime(ts)
                    for ts in timestamps_hist_np
                ]
            else:
                 timestamps_hist = timestamps_hist_np
            prices_hist = item_df.select("buyprice").tail(plot_points).to_numpy().flatten()
            plt.plot(timestamps_hist, prices_hist, label='Historical Price', color='blue')
            future_timestamps = [p["timestamp"] for p in vecm_preds]
            future_prices = [p["predicted_price"] for p in vecm_preds]
            if len(timestamps_hist) > 0 and len(future_timestamps) > 0:
                conn_hist_ts = timestamps_hist[-1]
                conn_pred_ts = future_timestamps[0]
                conn_hist_price = prices_hist[-1]
                conn_pred_price = future_prices[0]
                plt.plot([conn_hist_ts, conn_pred_ts], [conn_hist_price, conn_pred_price],
                         color='purple', linestyle='--', alpha=0.7)
            plt.plot(future_timestamps, future_prices, linestyle='--', marker='o',
                     label='Predicted Price (VECM)', color='purple')
            plt.xlabel('Timestamp')
            plt.ylabel('Price')
            plt.title(f"Price Prediction for {item_id} (VECM Model)")
            plt.legend()
            plt.grid(True)
            plot_dir = "integrated_plots"
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"{item_id}.png")
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Saved VECM plot for {item_id} to {plot_path}")
            # --- End Plot VECM Only ---
            logging.info(f"Finished processing item: {item_id} (VECM)")
        except Exception as e:
            logging.error(f"Error saving/plotting VECM results for {item_id}: {e}")
        # --- End Save VECM Results ---
        return # Move to next item if VECM was successful
    else:
        logging.info(f"VECM failed or not suitable for {item_id} ({vecm_msg}). Falling back to ARIMA-LSTM ensemble.")
    # --- End VECM Attempt ---
    # --- 3. If VECM Fails, Run ARIMA and LSTM Ensemble ---
    # --- Run ARIMA ---
    # Pass the 'periods' argument
    arima_success, arima_preds, arima_info, arima_msg, arima_last_price, arima_last_ts = \
        process_item_arima_primary(df, item_id, periods=periods, train_split_ratio=0.8) # Process item with ARIMA
    # --- Run LSTM ---
    # Pass the 'periods' argument
    lstm_success, lstm_preds, lstm_metrics, lstm_msg, lstm_last_price, lstm_last_ts = \
        process_item_lstm_primary(df, item_id, periods=periods, seq_length=24, epochs=50) # Process item with LSTM
    # --- Determine last price/timestamp (should be the same) ---
    last_price = None
    last_timestamp = None
    # Prefer ARIMA's if it ran, otherwise LSTM's
    if arima_success:
        last_price = arima_last_price
        last_timestamp = arima_last_ts
    elif lstm_success:
        last_price = lstm_last_price
        last_timestamp = lstm_last_ts
    if last_price is None or last_timestamp is None:
        logging.warning(f"Could not determine last price/timestamp for {item_id}. Skipping.")
        return
    # --- Combine Predictions if Both Succeeded ---
    ensemble_preds = None
    if arima_success and lstm_success:
        ensemble_preds = combine_predictions_simple_average(arima_preds, lstm_preds, item_id) # Combine ARIMA and LSTM predictions
        if ensemble_preds is None:
            logging.error(f"Failed to combine predictions for {item_id}.")
    else:
        logging.info(f"Only one model succeeded for {item_id}. No ensemble created.")
        if arima_success:
            logging.info(f"  - Using ARIMA predictions.")
        elif lstm_success:
            logging.info(f"  - Using LSTM predictions.")
        else:
            logging.warning(f"Both ARIMA and LSTM failed for {item_id}. Skipping.")
            return
    # --- Save Results and Plot ---
    try:
        pred_dir = "integrated_predictions"
        os.makedirs(pred_dir, exist_ok=True)
        pred_path = os.path.join(pred_dir, f"{item_id}.json")
        # --- Determine best model type for display ---
        best_model_type = "Ensemble" if ensemble_preds else ("ARIMA" if arima_success else "LSTM")
        # ---------------------------------------------
        predictions_data = {
            "item_id": item_id,
            "current_price": last_price,
            "current_timestamp": last_timestamp.isoformat(),
            "best_model_type": best_model_type,
            "models": {}
        }
        if arima_success:
            predictions_data["models"]["ARIMA"] = {
                "info": arima_info, # Contains AIC, in-sample MAE, R2
                "future_predictions": [
                    {"timestamp": p["timestamp"].isoformat(), "predicted_price": float(p["predicted_price"])}
                    for p in arima_preds
                ]
            }
        if lstm_success:
             predictions_data["models"]["LSTM"] = {
                "performance": lstm_metrics, # Contains train MAE, R2, val loss
                "future_predictions": [
                    {"timestamp": p["timestamp"].isoformat(), "predicted_price": float(p["predicted_price"])}
                    for p in lstm_preds
                ]
            }
        if ensemble_preds is not None:
            predictions_data["models"]["Ensemble"] = {
                "method": "Simple Average (50/50)",
                "future_predictions": [
                    {"timestamp": p["timestamp"].isoformat(), "predicted_price": float(p["predicted_price"])}
                    for p in ensemble_preds
                ]
            }
        with open(pred_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        logging.info(f"Saved ensemble results for {item_id} to {pred_path}")
        # --- Plotting ---
        plot_predictions_combined(
            df=item_df, # Pass the filtered df for historical data
            arima_preds=arima_preds if arima_success else [],
            lstm_preds=lstm_preds if lstm_success else [],
            ensemble_preds=ensemble_preds if ensemble_preds else [],
            vecm_preds=[], # No VECM preds in this path
            item_id=item_id
        )
        # --- End Plotting ---
        logging.info(f"Finished processing item: {item_id} (Ensemble/ARIMA/LSTM)")
    except Exception as e:
        logging.error(f"Error saving/plotting final results for {item_id}: {e}")
    # --- End Save Results and Plot ---

# --- Command-Line Interface (CLI) Entry Point ---
# Defines and parses command-line arguments, then calls the main processing function.
def main_cli_full():
    parser = argparse.ArgumentParser(description="Predict Hypixel Bazaar buy prices using integrated models (VECM, ARIMA, LSTM).")
    parser.add_argument("item_id", help="The item ID to predict (e.g., ENCHANTED_DIAMOND).")
    parser.add_argument("--data-file", default="bazaar_data.jsonl", help="Path to the input NDJSON data file (default: bazaar_data.jsonl).")
    parser.add_argument("--periods", type=int, default=24, help="Number of future periods (hours) to predict (default: 24).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()
    # Adjust logging level based on the --debug flag.
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    # Validate periods argument
    if args.periods <= 0:
        logging.error(f"Invalid number of periods: {args.periods}. Must be a positive integer.")
        return
    try:
        if not os.path.exists(args.data_file):
            logging.error(f"Data file '{args.data_file}' not found.")
            return
        df = load_data(args.data_file) # Load data from file
        # Pass the 'periods' argument to the processing function
        process_single_item_full(df, args.item_id, periods=args.periods) # Process the specified item
        logging.info(f"Integrated Model CLI Full Processing complete for item: {args.item_id} for {args.periods} periods!")
        logging.info(f"Results saved to integrated_predictions")
        logging.info(f"Models saved to integrated_models")
        logging.info(f"Plots saved to integrated_plots")
    except Exception as e:
        logging.error(f"Unexpected error in main: {e}")
        logging.debug(traceback.format_exc())

# --- Script Execution Entry Point ---
# Ensures the main function is called only when the script is executed directly.
if __name__ == "__main__":
    main_cli_full()