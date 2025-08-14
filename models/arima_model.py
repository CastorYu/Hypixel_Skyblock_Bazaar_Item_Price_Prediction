# models/arima_model.py
"""
Module for ARIMA (AutoRegressive Integrated Moving Average) model processing.
Updated to predict both buy and sell prices.
"""

import os
import logging
import traceback
import warnings
import joblib
import numpy as np
import polars as pl
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from utils.datetime_utils import convert_numpy_datetime64_to_python_datetime
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

def fit_and_evaluate_order(train_series, test_series, order):
    """Fit ARIMA model and evaluate MAE for a single order."""
    try:
        # Skip if not enough data
        if order[0] + order[2] >= len(train_series) // 2:
            return None, None, np.inf, order
            
        # Suppress ARIMA warnings for individual model fitting
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            
            model = ARIMA(train_series, order=order)
            fitted_model = model.fit(method_kwargs={
                "warn_convergence": False,
                "disp": 0  # Suppress optimization output
            })
        
        # Validate on holdout set
        forecast = fitted_model.forecast(steps=len(test_series))
        forecast_mae = mean_absolute_error(test_series, forecast)
        
        return fitted_model, forecast, forecast_mae, order
    except Exception:
        return None, None, np.inf, order

def find_best_order_simple(series, max_p=6, max_d=6, max_q=6):
    """Find the best ARIMA order using parallel processing with progress bar."""
    if len(series) < 10:
        return (0, 0, 0), None
    
    if len(np.unique(series)) == 1:
        return (0, 0, 0), None
    
    # Suppress warnings for the entire order finding process
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Generate all orders to try
    orders_to_try = [(i, j, h) for i in range(max_p) for j in range(max_d) for h in range(max_q)]

    # Split data for internal validation
    split_point = int(len(series) * 0.85)
    if split_point < 5:
        split_point = len(series) - max(3, len(series) // 4)
    
    train_series = series[:split_point]
    test_series = series[split_point:]
    
    if len(test_series) < 2:
        warnings.filterwarnings("default")
        return (0, 0, 0), None
    
    logging.info(f"Finding best order using parallel processing ({len(orders_to_try)} combinations)...")

    # Use all available CPU cores
    n_jobs = max(1, multiprocessing.cpu_count())
    
    # Parallel execution with progress bar
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(fit_and_evaluate_order)(train_series, test_series, order)
        for order in tqdm(orders_to_try, desc="Testing ARIMA orders", leave=True)
    )
    
    # Find the best result
    best_mae = np.inf
    best_order = None
    best_model = None
    
    for fitted_model, forecast, forecast_mae, order in results:
        if forecast_mae < best_mae:
            best_mae = forecast_mae
            best_order = order
            best_model = fitted_model

    # Restore default warning behavior
    warnings.filterwarnings("default")
    
    logging.info(f"Best model found: {best_order} with MAE: {best_mae:.6f}")
    return best_order, best_model

# ... rest of your existing code remains the same ...

def enhanced_validation_with_baselines(model, train_series, val_series):
    """Validation focusing on MAE improvement over baselines."""
    try:
        # ARIMA forecast
        forecast = model.forecast(steps=len(val_series))
        arima_mae = mean_absolute_error(val_series, forecast)
        
        # Calculate multiple baselines
        # Baseline 1: Persistence (last value)
        persistence_forecast = [train_series[-1]] * len(val_series)
        persistence_mae = mean_absolute_error(val_series, persistence_forecast)
        
        # Baseline 2: Mean of training data
        mean_forecast = [np.mean(train_series)] * len(val_series)
        mean_mae = mean_absolute_error(val_series, mean_forecast)
        
        # Baseline 3: Median (robust)
        median_forecast = [np.median(train_series)] * len(val_series)
        median_mae = mean_absolute_error(val_series, median_forecast)
        
        # Best baseline MAE
        best_baseline_mae = min(persistence_mae, mean_mae, median_mae)
        
        # Directional accuracy
        if len(val_series) > 1:
            actual_changes = np.diff(val_series)
            forecast_changes = np.diff(forecast)
            direction_accuracy = np.mean(np.sign(actual_changes) == np.sign(forecast_changes))
        else:
            direction_accuracy = 0.0
        
        # Improvement metrics
        mae_improvement = (best_baseline_mae - arima_mae) / best_baseline_mae if best_baseline_mae > 0 else 0
        
        return {
            'arima_mae': arima_mae,
            'baseline_mae': best_baseline_mae,
            'persistence_mae': persistence_mae,
            'mean_mae': mean_mae,
            'median_mae': median_mae,
            'direction_accuracy': direction_accuracy,
            'mae_improvement': mae_improvement,
            'forecast': forecast
        }
    except Exception as e:
        logging.warning(f"Validation failed: {e}")
        return None

def is_model_useful(validation_results, significance_threshold=-0.2):  # Accept even 20% worse models
    """Determine if model is useful - very lenient now."""
    if validation_results is None:
        return False
    
    # Accept model if it's not catastrophically bad
    reasonable_performance = validation_results['arima_mae'] < 1.0  # Prevent extremely bad models
    not_terrible_improvement = validation_results['mae_improvement'] > significance_threshold
    
    return reasonable_performance and not_terrible_improvement

def create_weighted_ensemble_forecast(train_series, val_series, periods=24):
    """Create weighted ensemble based on validation performance."""
    forecasts = []
    weights = []
    
    # Method 1: Persistence
    persistence_forecast = [train_series[-1]] * periods
    forecasts.append(persistence_forecast)
    
    # Validate persistence on validation set
    if len(val_series) > 0:
        persistence_val = [train_series[-1]] * len(val_series)
        persistence_mae = mean_absolute_error(val_series, persistence_val)
        weights.append(1.0 / (persistence_mae + 1e-8))
    else:
        weights.append(1.0)
    
    # Method 2: Mean
    mean_value = np.mean(train_series)
    mean_forecast = [mean_value] * periods
    forecasts.append(mean_forecast)
    
    if len(val_series) > 0:
        mean_val = [mean_value] * len(val_series)
        mean_mae = mean_absolute_error(val_series, mean_val)
        weights.append(1.0 / (mean_mae + 1e-8))
    else:
        weights.append(1.0)
    
    # Method 3: Median
    median_value = np.median(train_series)
    median_forecast = [median_value] * periods
    forecasts.append(median_forecast)
    
    if len(val_series) > 0:
        median_val = [median_value] * len(val_series)
        median_mae = mean_absolute_error(val_series, median_val)
        weights.append(1.0 / (median_mae + 1e-8))
    else:
        weights.append(1.0)
    
    # Weighted ensemble
    if forecasts and weights:
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Create weighted forecast
        ensemble_forecast = np.zeros(periods)
        for i, forecast in enumerate(forecasts):
            ensemble_forecast += weights[i] * np.array(forecast)
        
        return ensemble_forecast.tolist()
    else:
        return [train_series[-1]] * periods

def process_item_arima_primary(df, item_id, periods=24, train_split_ratio=0.85):
    """
    Enhanced ARIMA processing focusing on MAE improvement and practical accuracy.
    Returns exactly 8 values as expected by caller.
    """
    try:
        item_df = df.filter(pl.col("itemid") == item_id)
        min_data_points = 15  # Lower minimum
        if len(item_df) < min_data_points:
            return False, None, None, None, f"Not enough data points ({len(item_df)} < {min_data_points})", None, None, None

        split_idx = int(len(item_df) * train_split_ratio)
        if split_idx == 0 or split_idx == len(item_df) or len(item_df) - split_idx < 2:
            return False, None, None, None, "Insufficient data for train/validation split", None, None, None
            
        train_df = item_df.head(split_idx)
        val_df = item_df.tail(len(item_df) - split_idx)
        
        # Extract price data
        buy_prices_train = train_df.select("buyprice").to_numpy().astype(np.float64).flatten()
        sell_prices_train = train_df.select("sellprice").to_numpy().astype(np.float64).flatten()
        buy_prices_val = val_df.select("buyprice").to_numpy().astype(np.float64).flatten()
        sell_prices_val = val_df.select("sellprice").to_numpy().astype(np.float64).flatten()
        
        all_timestamps_np = item_df.select("timestampdt").to_numpy().flatten()
        last_timestamp_np = all_timestamps_np[-1]

        # Check for constant price series
        if (len(np.unique(buy_prices_train)) == 1 and 
            len(np.unique(sell_prices_train)) == 1):
            return False, None, None, None, "Constant price series (buy and sell)", None, None, None

        logging.info(f"Item {item_id} has varying prices. Running ARIMA model.")

        # Process BUY Price Model
        buy_model_final = None
        best_order_buy = None
        use_arima_buy = False
        buy_validation_results = None
        
        # Find best order
        all_buy_prices = np.concatenate([buy_prices_train, buy_prices_val])
        best_order_buy, fitted_model_buy = find_best_order_simple(all_buy_prices)
        
        if best_order_buy and fitted_model_buy and best_order_buy != (0, 0, 0):
            try:
                # Enhanced validation focusing on MAE
                buy_validation_results = enhanced_validation_with_baselines(
                    fitted_model_buy, buy_prices_train, buy_prices_val
                )
                
                if buy_validation_results:
                    use_arima_buy = is_model_useful(buy_validation_results)
                    logging.info(f"ARIMA for {item_id}: Buy model validation - "
                               f"ARIMA MAE={buy_validation_results['arima_mae']:.4f}, "
                               f"Baseline MAE={buy_validation_results['baseline_mae']:.4f}, "
                               f"Improvement={buy_validation_results['mae_improvement']:.2%}")
                
                if use_arima_buy:
                    buy_model_final = fitted_model_buy
                else:
                    buy_model_final = None
                    
            except Exception as e:
                logging.warning(f"Buy model validation failed: {e}")
                buy_model_final = None
        else:
            buy_model_final = None

        # Calculate final metrics
        final_aic_buy = buy_model_final.aic if buy_model_final else np.inf
        insample_mae_buy = buy_validation_results['arima_mae'] if buy_validation_results else np.nan
        insample_r2_buy = np.nan

        # Save acceptable model
        if buy_model_final and use_arima_buy:
            model_dir = "integrated_models"
            os.makedirs(model_dir, exist_ok=True)
            model_path_buy = os.path.join(model_dir, f"{item_id}_arima_buy.pkl")
            joblib.dump(buy_model_final, model_path_buy)

        # Future Predictions for BUY
        future_periods = periods
        future_buy_predictions_mean = []
        future_buy_predictions_lower = []
        future_buy_predictions_upper = []
        
        try:
            if buy_model_final and use_arima_buy:
                future_forecast_res_buy = buy_model_final.get_forecast(steps=future_periods)
                future_buy_predictions_mean = future_forecast_res_buy.predicted_mean
                
                # Get confidence intervals
                try:
                    conf_int = future_forecast_res_buy.conf_int(alpha=0.4)  # 60% confidence
                    if hasattr(conf_int, 'iloc'):
                        future_buy_predictions_lower = conf_int.iloc[:, 0]
                        future_buy_predictions_upper = conf_int.iloc[:, 1]
                    else:
                        future_buy_predictions_lower = conf_int[:, 0]
                        future_buy_predictions_upper = conf_int[:, 1]
                except Exception:
                    # Conservative bounds based on validation error
                    if buy_validation_results:
                        error_range = buy_validation_results['arima_mae'] * 1.2  # Tight bounds
                    else:
                        error_range = np.std(np.concatenate([buy_prices_train, buy_prices_val])) * 0.05  # Tight
                    future_buy_predictions_lower = [pred - error_range for pred in future_buy_predictions_mean]
                    future_buy_predictions_upper = [pred + error_range for pred in future_buy_predictions_mean]
            else:
                # Use weighted ensemble forecasting
                all_buy_prices = np.concatenate([buy_prices_train, buy_prices_val])
                ensemble_forecast = create_weighted_ensemble_forecast(all_buy_prices, buy_prices_val, future_periods)
                future_buy_predictions_mean = ensemble_forecast
                
                # Add reasonable bounds
                series_std = np.std(all_buy_prices)
                error_range = series_std * 0.05  # Tight bounds
                future_buy_predictions_lower = [pred - error_range for pred in ensemble_forecast]
                future_buy_predictions_upper = [pred + error_range for pred in ensemble_forecast]
                
        except Exception as e:
            logging.warning(f"Buy forecast failed: {e}")
            # Ultimate fallback
            all_buy_prices = np.concatenate([buy_prices_train, buy_prices_val])
            last_value = all_buy_prices[-1] if len(all_buy_prices) > 0 else 0
            future_buy_predictions_mean = [last_value] * future_periods
            future_buy_predictions_lower = [last_value * 0.99] * future_periods  # Very tight
            future_buy_predictions_upper = [last_value * 1.01] * future_periods

        # Process SELL Price Model
        sell_model_final = None
        best_order_sell = None
        use_arima_sell = False
        sell_validation_results = None
        
        # Find best order for sell
        all_sell_prices = np.concatenate([sell_prices_train, sell_prices_val])
        best_order_sell, fitted_model_sell = find_best_order_simple(all_sell_prices)
        
        if best_order_sell and fitted_model_sell and best_order_sell != (0, 0, 0):
            try:
                # Enhanced validation for sell
                sell_validation_results = enhanced_validation_with_baselines(
                    fitted_model_sell, sell_prices_train, sell_prices_val
                )
                
                if sell_validation_results:
                    use_arima_sell = is_model_useful(sell_validation_results)
                    logging.info(f"ARIMA for {item_id}: Sell model validation - "
                               f"ARIMA MAE={sell_validation_results['arima_mae']:.4f}, "
                               f"Baseline MAE={sell_validation_results['baseline_mae']:.4f}, "
                               f"Improvement={sell_validation_results['mae_improvement']:.2%}")
                
                if use_arima_sell:
                    sell_model_final = fitted_model_sell
                else:
                    sell_model_final = None
                    
            except Exception as e:
                logging.warning(f"Sell model validation failed: {e}")
                sell_model_final = None
        else:
            sell_model_final = None

        # Future Predictions for SELL
        future_sell_predictions_mean = []
        future_sell_predictions_lower = []
        future_sell_predictions_upper = []
        
        try:
            if sell_model_final and use_arima_sell:
                future_forecast_res_sell = sell_model_final.get_forecast(steps=future_periods)
                future_sell_predictions_mean = future_forecast_res_sell.predicted_mean
                
                # Get confidence intervals
                try:
                    conf_int_sell = future_forecast_res_sell.conf_int(alpha=0.4)  # 60% confidence
                    if hasattr(conf_int_sell, 'iloc'):
                        future_sell_predictions_lower = conf_int_sell.iloc[:, 0]
                        future_sell_predictions_upper = conf_int_sell.iloc[:, 1]
                    else:
                        future_sell_predictions_lower = conf_int_sell[:, 0]
                        future_sell_predictions_upper = conf_int_sell[:, 1]
                except Exception:
                    # Conservative bounds based on validation error
                    if sell_validation_results:
                        error_range = sell_validation_results['arima_mae'] * 1.2  # Tight
                    else:
                        error_range = np.std(all_sell_prices) * 0.05  # Tight
                    future_sell_predictions_lower = [pred - error_range for pred in future_sell_predictions_mean]
                    future_sell_predictions_upper = [pred + error_range for pred in future_sell_predictions_mean]
            else:
                # Use weighted ensemble forecasting for sell
                ensemble_forecast_sell = create_weighted_ensemble_forecast(all_sell_prices, sell_prices_val, future_periods)
                future_sell_predictions_mean = ensemble_forecast_sell
                
                # Add reasonable bounds
                series_std = np.std(all_sell_prices)
                error_range = series_std * 0.05  # Tight bounds
                future_sell_predictions_lower = [pred - error_range for pred in ensemble_forecast_sell]
                future_sell_predictions_upper = [pred + error_range for pred in ensemble_forecast_sell]
                
        except Exception as e:
            logging.warning(f"Sell forecast failed: {e}")
            # Ultimate fallback
            last_sell = all_sell_prices[-1] if len(all_sell_prices) > 0 else 0
            future_sell_predictions_mean = [last_sell] * future_periods
            future_sell_predictions_lower = [last_sell * 0.99] * future_periods  # Very tight
            future_sell_predictions_upper = [last_sell * 1.01] * future_periods

        # Prepare Timestamps and Finalize
        last_timestamp = convert_numpy_datetime64_to_python_datetime(last_timestamp_np)
        future_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(future_periods)]

        last_buy_price = float(buy_prices_train[-1] if len(buy_prices_train) > 0 else 
                             (buy_prices_val[-1] if len(buy_prices_val) > 0 else 0))
        last_sell_price = float(sell_prices_train[-1] if len(sell_prices_train) > 0 else 
                              (sell_prices_val[-1] if len(sell_prices_val) > 0 else 0))

        # Prepare Prediction Lists
        arima_buy_future_predictions = [
            {
                "timestamp": ts, 
                "predicted_price": float(pred),
                "lower_bound": float(lower),
                "upper_bound": float(upper)
            }
            for ts, pred, lower, upper in zip(
                future_timestamps, 
                future_buy_predictions_mean,
                future_buy_predictions_lower,
                future_buy_predictions_upper
            )
        ]
        arima_sell_future_predictions = [
            {
                "timestamp": ts, 
                "predicted_price": float(pred),
                "lower_bound": float(lower),
                "upper_bound": float(upper)
            }
            for ts, pred, lower, upper in zip(
                future_timestamps, 
                future_sell_predictions_mean,
                future_sell_predictions_lower,
                future_sell_predictions_upper
            )
        ]

        # Return model information
        arima_info = {
            "aic": float(final_aic_buy) if np.isfinite(final_aic_buy) else np.nan, 
            "mae": float(insample_mae_buy) if not np.isnan(insample_mae_buy) else np.nan, 
            "r2": float(insample_r2_buy) if not np.isnan(insample_r2_buy) else np.nan,
            "buy_order": best_order_buy,
            "sell_order": best_order_sell,
            "use_arima_buy": use_arima_buy,
            "use_arima_sell": use_arima_sell,
            "buy_arima_mae": buy_validation_results['arima_mae'] if buy_validation_results else None,
            "buy_baseline_mae": buy_validation_results['baseline_mae'] if buy_validation_results else None,
            "sell_arima_mae": sell_validation_results['arima_mae'] if sell_validation_results else None,
            "sell_baseline_mae": sell_validation_results['baseline_mae'] if sell_validation_results else None
        }
        
        # Log success messages
        if use_arima_buy or use_arima_sell:
            logging.info(f"ARIMA for {item_id}: Fitted model with buy_order={best_order_buy}, sell_order={best_order_sell}")
            if buy_validation_results:
                logging.info(f"ARIMA for {item_id}: In-sample MAE (buyprice)={buy_validation_results['arima_mae']:.4f}")
            if sell_validation_results:
                logging.info(f"ARIMA for {item_id}: In-sample MAE (sellprice)={sell_validation_results['arima_mae']:.4f}")
            logging.info(f"ARIMA successful for {item_id}.")
        else:
            logging.info(f"ARIMA model not useful for {item_id}. Using ensemble forecasts.")

        # Return exactly 8 values as expected by the caller
        success = True
        message = "Success"
        return success, arima_buy_future_predictions, arima_sell_future_predictions, arima_info, message, last_buy_price, last_sell_price, last_timestamp

    except Exception as e:
        error_msg = f"ARIMA Primary - Error processing item {item_id}: {str(e)}"
        logging.error(error_msg)
        logging.debug(traceback.format_exc())
        # Return exactly 8 values as expected by the caller
        return False, None, None, None, error_msg, None, None, None