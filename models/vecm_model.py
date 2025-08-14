# models/vecm_model.py
"""
Module for VECM (Vector Error Correction Model) processing.
Updated to return predictions for both buy and sell prices.
Handles statsmodels VECM dependencies gracefully.
"""

import os
import logging
import traceback
import warnings
import joblib
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import timedelta
from itertools import product
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

try:
    from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
    VECM_AVAILABLE = True
except ImportError:
    VECM_AVAILABLE = False
    logging.warning("statsmodels VECM not available. VECM models will be disabled.")
    VECM = None
    select_coint_rank = None

from utils.data_processing import is_price_constant
from utils.datetime_utils import convert_numpy_datetime64_to_python_datetime

def _prepare_data_for_vecm(data_df, item_id):
    """Prepare and preprocess data for VECM modeling"""
    smoothed_df = data_df.copy()
    
    for col in data_df.columns:
        if 'price' in col.lower():
            series = pl.Series(col, data_df[col].values)
            smoothed = series.rolling_mean(3).fill_null(strategy="forward").fill_null(strategy="backward")
            smoothed_df[col] = smoothed.to_pandas()
    
    return smoothed_df

def process_item_vecm(df, item_id, periods=24, max_lag=5, max_coint_rank=5, train_split_ratio=0.95, 
                      param_combinations=None, show_progress=True, max_workers=None):
    """
    Wrapper to process item with VECM using multiple features and multi-threading.
    Returns: (success_flag, buy_predictions_list, sell_predictions_list, metrics_dict, message, last_buy_price, last_sell_price, last_timestamp)
    """
    if not VECM_AVAILABLE or VECM is None or select_coint_rank is None:
        return False, None, None, None, None, None, None, None, "VECM not available"
    
    # Define comprehensive parameter combinations (255 total) with FIXED train_split_ratio
    if param_combinations is None:
        param_combinations = [
            {'max_lag': lag, 'max_coint_rank': rank, 'train_split_ratio': train_split_ratio}  # Fixed ratio
            for lag, rank in product(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],      # 15 values
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # 17 values
            )
        ]
    
    # Set default number of workers
    if max_workers is None:
        max_workers = min(8, max(1, mp.cpu_count() - 1))  # Leave one CPU free
    
    try:
        item_df = df.filter(pl.col("itemid") == item_id)
        min_data_points = 25
        if len(item_df) < min_data_points:
            return False, None, None, None, None, None, None, None, f"Not enough data points ({len(item_df)} < {min_data_points})"

        # Select relevant features
        feature_columns = ['buyprice', 'sellprice', 'buyvolume', 'sellvolume']
        available_features = []
        for col in feature_columns:
            if col in item_df.columns:
                col_data = item_df.select(col).to_numpy().astype(np.float64).flatten()
                if np.var(col_data) > 1e-10:
                     available_features.append(col)
                else:
                     logging.info(f"VECM for {item_id}: Feature {col} has zero variance, skipping.")
            else:
                 logging.info(f"VECM for {item_id}: Feature {col} not found, skipping.")
        if len(available_features) < 2:
            return False, None, None, None, None, None, None, None, f"Not enough valid features with variance ({len(available_features)} < 2)"
        data_df = item_df.select(available_features).to_pandas()
        
        # Data preprocessing
        processed_data_df = _prepare_data_for_vecm(data_df, item_id)

        # Check for constant series
        constant_series_count = 0
        for col in processed_data_df.columns:
            if is_price_constant(processed_data_df[col].values, rel_tol=1e-4):
                 constant_series_count += 1
        if constant_series_count == len(processed_data_df.columns):
             return False, None, None, None, None, None, None, None, "All VECM features are constant"

        # Initialize best model tracking
        best_model_info = {
            'model': None,
            'metrics': {'insample_mae': np.inf, 'insample_r2': -np.inf},
            'predictions': None,
            'params': None
        }
        
        # Multi-threaded parameter testing
        results = []
        
        # Process parameters in chunks to manage memory and show progress
        chunk_size = max(1, len(param_combinations) // max_workers)
        param_chunks = [param_combinations[i:i + chunk_size] 
                       for i in range(0, len(param_combinations), chunk_size)]
        
        overall_pbar = tqdm(total=len(param_combinations), 
                           desc=f"VECM {item_id}", 
                           disable=not show_progress)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(_process_param_chunk, 
                              item_df, item_id, processed_data_df, available_features, 
                              periods, chunk, show_progress): chunk 
                for chunk in param_chunks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_results = future.result()
                results.extend(chunk_results)
                
                # Update progress bar
                overall_pbar.update(len(future_to_chunk[future]))
                
                # Update best model if needed
                for result in chunk_results:
                    if result['success'] and result['metrics']:
                        current_mae = result['metrics'].get('insample_mae', np.inf)
                        if not np.isnan(current_mae) and current_mae < best_model_info['metrics']['insample_mae']:
                            best_model_info = {
                                'model': result['model_data'],
                                'metrics': result['metrics'],
                                'predictions': result['predictions'],
                                'params': result['params'],
                                'message': result['message']
                            }
                            if show_progress:
                                overall_pbar.set_postfix({
                                    'best_mae': f"{current_mae:.4f}",
                                    'params': f"L{result['params']['max_lag']}_R{result['params']['max_coint_rank']}"
                                })
        
        overall_pbar.close()
        
        # If no successful model was found, return failure
        if best_model_info['model'] is None:
            return False, None, None, None, None, None, None, None, "No successful model configuration found"
        
        # Extract best results
        buy_preds, sell_preds, last_buy, last_sell, last_ts = best_model_info['model']
        metrics = best_model_info['metrics']
        best_params = best_model_info['params']
        
        logging.info(f"VECM for {item_id}: Best params - Lag:{best_params['max_lag']}, Rank:{best_params['max_coint_rank']}")
        
        return True, buy_preds, sell_preds, metrics, "Success", last_buy, last_sell, last_ts
        
    except Exception as e:
        error_msg = f"VECM Primary - Error processing item {item_id}: {str(e)}"
        logging.error(error_msg)
        logging.debug(traceback.format_exc())
        return False, None, None, None, None, None, None, None, error_msg

def _process_param_chunk(item_df, item_id, processed_data_df, available_features, periods, 
                        param_chunk, show_progress):
    """Process a chunk of parameters in a single thread"""
    chunk_results = []
    
    for params in param_chunk:
        try:
            success, buy_preds, sell_preds, metrics, msg, last_buy, last_sell, last_ts = _process_with_params(
                item_df, item_id, processed_data_df, available_features, periods, 
                params['max_lag'], params['max_coint_rank'], params['train_split_ratio']
            )
            
            chunk_results.append({
                'success': success,
                'model_data': (buy_preds, sell_preds, last_buy, last_sell, last_ts) if success else None,
                'metrics': metrics,
                'predictions': (buy_preds, sell_preds) if success else None,
                'params': params,
                'message': msg
            })
            
        except Exception as e:
            logging.debug(f"VECM for {item_id}: Failed with params {params}: {str(e)}")
            chunk_results.append({
                'success': False,
                'model_data': None,
                'metrics': None,
                'predictions': None,
                'params': params,
                'message': str(e)
            })
    
    return chunk_results

def _process_with_params(item_df, item_id, processed_data_df, available_features, periods, 
                        max_lag, max_coint_rank, train_split_ratio):
    """Process VECM with specific parameters"""
    
    # Use maximum available data for training
    split_idx = int(len(processed_data_df) * train_split_ratio)
    train_data = processed_data_df.iloc[:split_idx]
    
    # Enhanced Cointegration Rank Selection
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            
            best_rank = 1
            best_criteria = np.inf
            
            for det_order in [0, 1]:
                for k_ar_diff in [0, max_lag]:
                    try:
                        coint_result = select_coint_rank(
                            train_data, 
                            det_order=det_order, 
                            k_ar_diff=k_ar_diff, 
                            method='trace', 
                            signif=0.05
                        )
                        
                        stability_score = coint_result.rank + (det_order * 0.5) + (k_ar_diff * 0.3)
                        
                        if stability_score < best_criteria:
                            best_criteria = stability_score
                            best_rank = coint_result.rank
                    except:
                        continue
            
            coint_rank = max(0, min(best_rank, max_coint_rank))
        logging.debug(f"VECM for {item_id}: Selected cointegration rank: {coint_rank}")
    except Exception as e:
        logging.warning(f"VECM for {item_id}: Could not determine coint rank (error: {e}). Using rank=1.")
        coint_rank = 1

    # Fit Multiple VECM Models and Select Best
    try:
        best_model = None
        best_ic = np.inf
        best_k_ar_diff = 1
        best_det = 'co'
        
        configs = [
            {'k_ar_diff': 1, 'deterministic': 'co'},
            {'k_ar_diff': 1, 'deterministic': 'ci'},
            {'k_ar_diff': 2, 'deterministic': 'co'},
            {'k_ar_diff': 1, 'deterministic': 'n'},
        ]
        
        for config in configs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    temp_model = VECM(
                        train_data, 
                        k_ar_diff=config['k_ar_diff'], 
                        coint_rank=coint_rank, 
                        deterministic=config['deterministic']
                    )
                    temp_fitted = temp_model.fit()
                    
                    if temp_fitted.aic < best_ic:
                        best_ic = temp_fitted.aic
                        best_model = temp_fitted
                        best_k_ar_diff = config['k_ar_diff']
                        best_det = config['deterministic']
            except:
                continue
        
        if best_model is not None:
            vecm_fitted = best_model
            k_ar_diff_used = best_k_ar_diff
            deterministic_used = best_det
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                k_ar_diff_used = 1
                deterministic_used = 'co'
                vecm_model = VECM(
                    train_data, 
                    k_ar_diff=k_ar_diff_used, 
                    coint_rank=coint_rank, 
                    deterministic=deterministic_used
                )
                vecm_fitted = vecm_model.fit()
                
    except Exception as e:
         return False, None, None, None, None, None, None, None, f"Failed to fit VECM model: {e}"
    logging.debug(f"VECM for {item_id}: Fitted model with k_ar_diff={k_ar_diff_used}, coint_rank={coint_rank}, deterministic={deterministic_used}")

    # Improved In-Sample Evaluation
    try:
        buyprice_col_index = list(train_data.columns).index('buyprice') if 'buyprice' in train_data.columns else None
        sellprice_col_index = list(train_data.columns).index('sellprice') if 'sellprice' in train_data.columns else None

        if buyprice_col_index is None:
            raise ValueError("buyprice column not found in VECM data.")

        insample_fitted = vecm_fitted.fittedvalues[:, buyprice_col_index]
        insample_actual_full = train_data['buyprice'].values
        
        n_fitted_obs = len(insample_fitted)
        if n_fitted_obs > len(insample_actual_full):
            raise ValueError(f"More fitted values ({n_fitted_obs}) than actual values ({len(insample_actual_full)})")
        
        insample_actual_aligned = insample_actual_full[-n_fitted_obs:]
        
        if len(insample_fitted) != len(insample_actual_aligned):
             raise ValueError(f"Length mismatch: fitted ({len(insample_fitted)}) vs actual ({len(insample_actual_aligned)})")

        insample_mae = mean_absolute_error(insample_actual_aligned, insample_fitted)
        if np.var(insample_actual_aligned) > 1e-10:
            insample_r2 = r2_score(insample_actual_aligned, insample_fitted)
        else:
            insample_r2 = np.nan
            logging.warning(f"VECM for {item_id}: In-sample actual buy values have near-zero variance, R2 set to NaN.")
    except Exception as e_eval:
        logging.error(f"VECM for {item_id}: Error during in-sample evaluation: {e_eval}")
        logging.debug(traceback.format_exc())
        insample_mae = np.nan
        insample_r2 = np.nan
    logging.debug(f"VECM for {item_id}: In-sample MAE (buyprice)={insample_mae:.4f}, R2={insample_r2:.4f}")

    # Save Model
    model_dir = "integrated_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{item_id}_vecm.pkl")
    joblib.dump(vecm_fitted, model_path)

    # Future Predictions with Bias Correction
    future_periods = periods
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            forecast_result = vecm_fitted.predict(steps=future_periods)

        # Extract both buyprice and sellprice forecasts
        future_buyprice_forecasts = forecast_result[:, buyprice_col_index] if buyprice_col_index is not None else [np.nan]*future_periods
        future_sellprice_forecasts = forecast_result[:, sellprice_col_index] if sellprice_col_index is not None else [np.nan]*future_periods

        if not np.isnan(insample_mae) and insample_mae > 0:
            dampening_factor = min(1.0, 0.9 + 0.1 * (1 - min(insample_mae, 5) / 5))
            if dampening_factor < 1.0:
                last_buy = processed_data_df['buyprice'].iloc[-1]
                future_buyprice_forecasts = [
                    pred * dampening_factor + last_buy * (1 - dampening_factor) 
                    for pred in future_buyprice_forecasts
                ]
                
    except Exception as e:
        return False, None, None, None, None, None, None, None, f"Failed to forecast with VECM: {e}"

    # Prepare Output
    timestamps_np = item_df.select("timestampdt").to_numpy().flatten()
    last_timestamp_np = timestamps_np[-1]
    if isinstance(last_timestamp_np, np.datetime64):
        last_timestamp = convert_numpy_datetime64_to_python_datetime(last_timestamp_np)
    else:
        last_timestamp = last_timestamp_np
    future_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(future_periods)]

    buy_prices_np = item_df.select("buyprice").to_numpy().astype(np.float32).flatten()
    sell_prices_np = item_df.select("sellprice").to_numpy().astype(np.float32).flatten()
    last_buy_price = float(buy_prices_np[-1]) if len(buy_prices_np) > 0 else np.nan
    last_sell_price = float(sell_prices_np[-1]) if len(sell_prices_np) > 0 else np.nan

    # Create prediction dictionaries for both buy and sell prices
    vecm_buy_future_predictions = [
        {"timestamp": ts, "predicted_price": float(pred)}
        for ts, pred in zip(future_timestamps, future_buyprice_forecasts)
    ]
    vecm_sell_future_predictions = [
        {"timestamp": ts, "predicted_price": float(pred)}
        for ts, pred in zip(future_timestamps, future_sellprice_forecasts)
    ]

    metrics = {
        "insample_mae": float(insample_mae) if not np.isnan(insample_mae) else None,
        "insample_r2": float(insample_r2) if not np.isnan(insample_r2) else None,
        "coint_rank": int(coint_rank),
        "k_ar_diff": int(k_ar_diff_used)
    }
    
    return True, vecm_buy_future_predictions, vecm_sell_future_predictions, metrics, "Success", last_buy_price, last_sell_price, last_timestamp