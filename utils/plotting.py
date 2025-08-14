# utils/plotting.py
"""
Module for plotting utilities for model predictions (Buy and Sell prices).
"""
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from utils.datetime_utils import convert_numpy_datetime64_to_python_datetime # Import utility

def plot_predictions_combined(df, arima_buy_preds, arima_sell_preds,
                              lstm_buy_preds, lstm_sell_preds,
                              ensemble_buy_preds, ensemble_sell_preds,
                              vecm_buy_preds, vecm_sell_preds, item_id):
    """
    Plots historical buy/sell data and predictions from all models.
    Assumes prediction lists contain dicts with 'timestamp' and 'predicted_price'.
    Filters out negative predicted prices before plotting.
    """
    try:
        plt.figure(figsize=(15, 7)) # Slightly wider figure

        # --- Helper Function to Filter Negative Predictions ---
        def filter_negative_predictions(pred_list):
            """Filters out predictions with negative prices."""
            if not pred_list:
                return pred_list
            filtered = [pred for pred in pred_list if pred.get("predicted_price", 0) >= 0]
            if len(filtered) != len(pred_list):
                logging.info(f"Filtered out {len(pred_list) - len(filtered)} negative predictions for {item_id}")
            return filtered
        # --- End Filter Helper ---

        # Apply filter to all prediction lists
        arima_buy_preds = filter_negative_predictions(arima_buy_preds)
        arima_sell_preds = filter_negative_predictions(arima_sell_preds)
        lstm_buy_preds = filter_negative_predictions(lstm_buy_preds)
        lstm_sell_preds = filter_negative_predictions(lstm_sell_preds)
        ensemble_buy_preds = filter_negative_predictions(ensemble_buy_preds)
        ensemble_sell_preds = filter_negative_predictions(ensemble_sell_preds)
        vecm_buy_preds = filter_negative_predictions(vecm_buy_preds)
        vecm_sell_preds = filter_negative_predictions(vecm_sell_preds)


        # --- Plot Historical Data ---
        plot_points = min(200, len(df))
        timestamps_hist_np = df.select("timestampdt").tail(plot_points).to_numpy().flatten()
        if len(timestamps_hist_np) > 0 and isinstance(timestamps_hist_np[0], np.datetime64):
            timestamps_hist = [convert_numpy_datetime64_to_python_datetime(ts) for ts in timestamps_hist_np]
        else:
             timestamps_hist = list(timestamps_hist_np)
        buy_prices_hist = df.select("buyprice").tail(plot_points).to_numpy().astype(np.float32).flatten()
        sell_prices_hist = df.select("sellprice").tail(plot_points).to_numpy().astype(np.float32).flatten()

        # Optional: Filter historical data for negative prices (less likely but possible)
        # This requires keeping track of indices where prices are non-negative
        # For simplicity here, we'll assume historical data is mostly valid
        # but you could add a check if needed.
        valid_hist_indices = np.where((buy_prices_hist >= 0) & (sell_prices_hist >= 0))[0]
        if len(valid_hist_indices) > 0:
            # Only plot valid historical points
            timestamps_hist = [timestamps_hist[i] for i in valid_hist_indices]
            buy_prices_hist = buy_prices_hist[valid_hist_indices]
            sell_prices_hist = sell_prices_hist[valid_hist_indices]
            plt.plot(timestamps_hist, buy_prices_hist, label='Historical Buy Price', color='blue', alpha=0.7)
            plt.plot(timestamps_hist, sell_prices_hist, label='Historical Sell Price', color='orange', alpha=0.7)
        else:
            logging.warning(f"No valid historical data points (non-negative) found for {item_id}. Skipping historical plot.")
        # --- End Historical Data ---


        # --- Helper for Plotting Predictions ---
        def plot_pred_lines(preds, label, color, linestyle='--', marker='o', is_sell=False):
            """Generic function to plot prediction lines."""
            if not preds:
                logging.info(f"No predictions to plot for {label} for {item_id}")
                return
            pred_timestamps_raw = [p["timestamp"] for p in preds]
            pred_timestamps = []
            for ts in pred_timestamps_raw:
                if isinstance(ts, str):
                    # Attempt to parse ISO format string
                    try:
                        pred_timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                    except ValueError:
                        logging.warning(f"Could not parse timestamp string: {ts}")
                        pred_timestamps.append(None)
                elif isinstance(ts, (datetime, np.datetime64)):
                    if isinstance(ts, np.datetime64):
                        pred_timestamps.append(convert_numpy_datetime64_to_python_datetime(ts))
                    else:
                        pred_timestamps.append(ts)
                else:
                    logging.warning(f"Unknown timestamp type: {type(ts)}")
                    pred_timestamps.append(None)

            # Filter out None values if any parsing failed
            valid_indices = [i for i, ts in enumerate(pred_timestamps) if ts is not None]
            if not valid_indices:
                logging.warning(f"No valid timestamps to plot for {label} for {item_id}")
                return

            pred_timestamps = [pred_timestamps[i] for i in valid_indices]
            pred_prices = [preds[i]["predicted_price"] for i in valid_indices] # These are already non-negative

            # Connect last historical point to first prediction point (if both exist and are valid)
            if len(timestamps_hist) > 0 and len(pred_timestamps) > 0:
                # Determine which historical price to connect from
                hist_price_to_use = sell_prices_hist[-1] if is_sell else buy_prices_hist[-1]
                # Ensure the historical price we connect from is also non-negative
                if hist_price_to_use >= 0:
                    hist_ts = timestamps_hist[-1]
                    pred_ts = pred_timestamps[0]
                    pred_price = pred_prices[0]
                    plt.plot([hist_ts, pred_ts], [hist_price_to_use, pred_price],
                            color=color, linestyle=linestyle, alpha=0.4, linewidth=1)

            plt.plot(pred_timestamps, pred_prices, linestyle=linestyle, marker=marker,
                     label=label, color=color, markersize=4, alpha=0.8)
        # --- End Helper ---


        # --- Plot Model Predictions ---
        # --- ARIMA ---
        plot_pred_lines(arima_buy_preds, 'ARIMA Buy Pred', 'blue', linestyle='-.', marker='s')
        plot_pred_lines(arima_sell_preds, 'ARIMA Sell Pred', 'orange', linestyle='-.', marker='s', is_sell=True)
        # --- LSTM ---
        plot_pred_lines(lstm_buy_preds, 'LSTM Buy Pred', 'cyan', linestyle=':', marker='^')
        plot_pred_lines(lstm_sell_preds, 'LSTM Sell Pred', 'gold', linestyle=':', marker='^', is_sell=True)
        # --- Ensemble ---
        plot_pred_lines(ensemble_buy_preds, 'Ensemble Buy Pred', 'green', linestyle='--', marker='d')
        plot_pred_lines(ensemble_sell_preds, 'Ensemble Sell Pred', 'red', linestyle='--', marker='d', is_sell=True)
        # --- VECM ---
        plot_pred_lines(vecm_buy_preds, 'VECM Buy Pred', 'purple', linestyle='-', marker='x')
        plot_pred_lines(vecm_sell_preds, 'VECM Sell Pred', 'brown', linestyle='-', marker='x', is_sell=True)
        # --- End Model Predictions ---


        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.title(f"Buy/Sell Price Prediction for {item_id}")
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1)) # Legend outside plot
        plt.grid(True, alpha=0.5)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend

        plot_dir = "integrated_plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{item_id}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved integrated buy/sell plot (filtered) for {item_id} to {plot_path}")
    except Exception as e:
        logging.error(f"Error plotting combined buy/sell predictions for {item_id}: {e}")
        import traceback
        logging.debug(traceback.format_exc())
