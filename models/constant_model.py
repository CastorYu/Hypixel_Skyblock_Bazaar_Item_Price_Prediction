# models/constant_model.py
"""
Module for handling constant price series using a naive prediction model.
Updated to predict both buy and sell prices.
"""

import os
import json
import logging
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import numpy as np
from utils.datetime_utils import convert_numpy_datetime64_to_python_datetime # Import utility

# Generates future predictions for constant buy and sell prices.
def predict_future_constant(last_buy_price: float, last_sell_price: float, last_timestamp: datetime, periods: int = 24):
    """
    Generates future predictions for constant buy and sell prices.
    Args:
        last_buy_price (float): The last observed buy price.
        last_sell_price (float): The last observed sell price.
        last_timestamp (datetime): The timestamp of the last observation.
        periods (int): Number of future periods to predict.
    Returns:
        tuple: A tuple containing two lists:
               - buy_predictions: List of dicts with 'timestamp' and 'predicted_price' for buy.
               - sell_predictions: List of dicts with 'timestamp' and 'predicted_price' for sell.
    """
    logging.debug(f"Predicting future prices: buy={last_buy_price}, sell={last_sell_price}, timestamp={last_timestamp}")
    
    buy_predictions = []
    sell_predictions = []
    for i in range(periods):
        next_timestamp = last_timestamp + timedelta(hours=i+1)
        buy_predictions.append({
            "timestamp": next_timestamp,
            "predicted_price": float(last_buy_price)
        })
        sell_predictions.append({
            "timestamp": next_timestamp,
            "predicted_price": float(last_sell_price)
        })
    return buy_predictions, sell_predictions

# Saves results and plots for a constant price series (buy and sell).
def save_and_plot_constant(item_id: str, last_buy_price: float, last_sell_price: float, last_timestamp: datetime,
                           buy_future_predictions: list, sell_future_predictions: list, item_df_for_plotting):
    """
    Saves results and plots for a constant price series (buy and sell).
    Args:
        item_id (str): The ID of the item.
        last_buy_price (float): The last observed buy price.
        last_sell_price (float): The last observed sell price.
        last_timestamp (datetime): The timestamp of the last observation.
        buy_future_predictions (list): List of buy price prediction dicts.
        sell_future_predictions (list): List of sell price prediction dicts.
        item_df_for_plotting (polars.DataFrame): DataFrame for plotting historical data.
    """
    try:
        pred_dir = "integrated_predictions"
        os.makedirs(pred_dir, exist_ok=True)
        pred_path = os.path.join(pred_dir, f"{item_id}.json")
        
        # Combine buy and sell predictions for saving
        combined_predictions = []
        if buy_future_predictions and sell_future_predictions and len(buy_future_predictions) == len(sell_future_predictions):
            for buy_pred, sell_pred in zip(buy_future_predictions, sell_future_predictions):
                 # Basic check for timestamp alignment (should always be true here)
                if buy_pred['timestamp'] != sell_pred['timestamp']:
                     logging.warning(f"Timestamp mismatch in constant model saving: {buy_pred['timestamp']} vs {sell_pred['timestamp']}")
                combined_predictions.append({
                    "timestamp": buy_pred["timestamp"].isoformat(), # Use buy timestamp
                    "buy_predicted_price": float(buy_pred["predicted_price"]),
                    "sell_predicted_price": float(sell_pred["predicted_price"])
                })
        else:
            # Fallback if lists are mismatched (shouldn't happen)
            logging.error("Mismatch in buy/sell prediction lists for constant model saving.")
            combined_predictions = []

        predictions_data = {
            "item_id": item_id,
            "current_buy_price": float(last_buy_price),
            "current_sell_price": float(last_sell_price),
            "current_timestamp": last_timestamp.isoformat(),
            "best_model_type": "Constant (Naive)",
            "models": {
                "Constant": {
                    "method": "Naive (Last Price Repeated for Buy/Sell)",
                    "performance": {"note": "Prices are constant, no training metrics"},
                    "future_predictions": combined_predictions
                }
            }
        }
        with open(pred_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        logging.info(f"Saved constant model results for {item_id} to {pred_path}")
        
        # --- Plotting ---
        plt.figure(figsize=(15, 7)) # Wider figure for legend
        
        # --- Plot Historical Data ---
        plot_points = min(200, len(item_df_for_plotting))
        timestamps_hist_np = item_df_for_plotting.select("timestampdt").tail(plot_points).to_numpy().flatten()
        if len(timestamps_hist_np) > 0 and isinstance(timestamps_hist_np[0], np.datetime64):
            timestamps_hist = [
                convert_numpy_datetime64_to_python_datetime(ts)
                for ts in timestamps_hist_np
            ]
        else:
             timestamps_hist = list(timestamps_hist_np)
        buy_prices_hist = item_df_for_plotting.select("buyprice").tail(plot_points).to_numpy().flatten()
        sell_prices_hist = item_df_for_plotting.select("sellprice").tail(plot_points).to_numpy().flatten()
        
        plt.plot(timestamps_hist, buy_prices_hist, label='Historical Buy Price', color='blue', alpha=0.7)
        plt.plot(timestamps_hist, sell_prices_hist, label='Historical Sell Price', color='orange', alpha=0.7)
        # --- End Historical Data ---
        
        # --- Plot Future Predictions ---
        if buy_future_predictions and sell_future_predictions:
            future_buy_timestamps = [p["timestamp"] for p in buy_future_predictions]
            future_buy_prices = [p["predicted_price"] for p in buy_future_predictions]
            future_sell_timestamps = [p["timestamp"] for p in sell_future_predictions]
            future_sell_prices = [p["predicted_price"] for p in sell_future_predictions]
            
            # Connect historical to future
            if len(timestamps_hist) > 0:
                conn_hist_ts = timestamps_hist[-1]
                conn_buy_hist_price = buy_prices_hist[-1]
                conn_sell_hist_price = sell_prices_hist[-1]
                if future_buy_timestamps:
                    conn_buy_pred_ts = future_buy_timestamps[0]
                    conn_buy_pred_price = future_buy_prices[0]
                    plt.plot([conn_hist_ts, conn_buy_pred_ts], [conn_buy_hist_price, conn_buy_pred_price],
                             color='blue', linestyle='--', alpha=0.5, linewidth=1)
                if future_sell_timestamps:
                    conn_sell_pred_ts = future_sell_timestamps[0]
                    conn_sell_pred_price = future_sell_prices[0]
                    plt.plot([conn_hist_ts, conn_sell_pred_ts], [conn_sell_hist_price, conn_sell_pred_price],
                             color='orange', linestyle='--', alpha=0.5, linewidth=1)
            
            # Plot future lines
            plt.plot(future_buy_timestamps, future_buy_prices, linestyle='--', marker='o',
                     label='Predicted Buy Price (Constant)', color='blue', alpha=0.8)
            plt.plot(future_sell_timestamps, future_sell_prices, linestyle='--', marker='o',
                     label='Predicted Sell Price (Constant)', color='orange', alpha=0.8)
        # --- End Future Predictions ---
        
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.title(f"Price Prediction for {item_id} (Constant Model)")
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1)) # Legend outside
        plt.grid(True, alpha=0.5)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
        
        plot_dir = "integrated_plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{item_id}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved constant model plot for {item_id} to {plot_path}")
        # --- End Plotting ---
        
    except Exception as e:
        logging.error(f"Error saving/plotting constant model results for {item_id}: {e}")