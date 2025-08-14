# main.py

"""
Main CLI script for integrated price prediction using VECM, ARIMA, and LSTM models.
Selection of the best model is based on In-Sample MAE for buy price.
Predictions for both buy and sell prices are displayed and plotted.
"""

import os,logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Standard Library Imports ---
import os
import logging
import argparse
import traceback
import sys
import warnings
import json
import polars as pl
import numpy as np

# Import termcolor for colored console output
try:
    from termcolor import colored
    TERMCOLOR_AVAILABLE = True
except ImportError:
    TERMCOLOR_AVAILABLE = False
    def colored(text, color=None, on_color=None, attrs=None):
        # Fallback function if termcolor is not available
        return text

# --- Custom JSON Encoder ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.datetime64, pl.Datetime)):
            return str(obj)
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return super(NpEncoder, self).default(obj)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cli.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- Windows Console Encoding Fix ---
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

logging.info("CLI Initiated successful.")

# --- Import Custom Modules ---
from utils.data_processing import load_data, is_price_constant
from utils.datetime_utils import convert_numpy_datetime64_to_python_datetime
from models.arima_model import process_item_arima_primary
from models.lstm_model import process_item_lstm_primary, combine_predictions_simple_average_single
from models.vecm_model import process_item_vecm
from utils.plotting import plot_predictions_combined

# --- Helper Function ---
def format_predictions_for_saving(buy_preds, sell_preds):
    """Combines separate buy/sell prediction lists into the JSON structure."""
    combined_preds = []
    if buy_preds and sell_preds and len(buy_preds) == len(sell_preds):
        for b, s in zip(buy_preds, sell_preds):
            # Ensure timestamps are ISO formatted strings
            b_ts = b["timestamp"].isoformat() if hasattr(b["timestamp"], 'isoformat') else str(b["timestamp"])
            s_ts = s["timestamp"].isoformat() if hasattr(s["timestamp"], 'isoformat') else str(s["timestamp"])
            # Basic check for timestamp alignment (optional but good)
            if b_ts != s_ts:
                logging.warning(f"Timestamp mismatch in formatting: {b_ts} vs {s_ts}")
            combined_preds.append({
                "timestamp": b_ts,
                "buy_predicted_price": float(b["predicted_price"]) if not np.isnan(b["predicted_price"]) else None,
                "sell_predicted_price": float(s["predicted_price"]) if not np.isnan(s["predicted_price"]) else None
            })
    elif buy_preds:
        logging.info("Formatting: Only buy predictions available for saving.")
        for b in buy_preds:
            b_ts = b["timestamp"].isoformat() if hasattr(b["timestamp"], 'isoformat') else str(b["timestamp"])
            combined_preds.append({
                "timestamp": b_ts,
                "buy_predicted_price": float(b["predicted_price"]) if not np.isnan(b["predicted_price"]) else None,
                "sell_predicted_price": None
            })
    elif sell_preds:
        logging.info("Formatting: Only sell predictions available for saving.")
        for s in sell_preds:
            s_ts = s["timestamp"].isoformat() if hasattr(s["timestamp"], 'isoformat') else str(s["timestamp"])
            combined_preds.append({
                "timestamp": s_ts,
                "buy_predicted_price": None,
                "sell_predicted_price": float(s["predicted_price"]) if not np.isnan(s["predicted_price"]) else None
            })
    else:
        logging.warning("Formatting: No predictions provided.")
    return combined_preds

# --- Main Processing Logic ---
def process_single_item_full(df, item_id, periods=24, graph=True):
    """Processes a single item, running all applicable models and saving results.
    Args:
        graph: If True, generates and saves plots
    """
    logging.info(f"Processing item: {item_id} for {periods} periods ahead")
    item_df = df.filter(pl.col("itemid") == item_id)
    if len(item_df) == 0:
        logging.error(f"Item ID '{item_id}' not found in the data.")
        return

    # --- 1. Check for Constant Price (Check buy price constancy) ---
    buy_prices_np = item_df.select("buyprice").to_numpy().astype(np.float32).flatten()
    sell_prices_np = item_df.select("sellprice").to_numpy().astype(np.float32).flatten()
    if is_price_constant(buy_prices_np) and is_price_constant(sell_prices_np):
        logging.info(f"Item {item_id} has constant/near-constant prices. Using Naive model.")
        last_buy_price = float(buy_prices_np[-1])
        last_sell_price = float(sell_prices_np[-1])
        timestamps_np = item_df.select("timestampdt").to_numpy().flatten()
        last_timestamp_np = timestamps_np[-1]
        if isinstance(last_timestamp_np, np.datetime64):
            last_timestamp = convert_numpy_datetime64_to_python_datetime(last_timestamp_np)
        else:
            last_timestamp = last_timestamp_np

        # --- Naive Predictions ---
        from models.constant_model import predict_future_constant
        # FIXED: Pass all required arguments including last_timestamp
        const_buy_preds, const_sell_preds = predict_future_constant(
            last_buy_price=last_buy_price,
            last_sell_price=last_sell_price,
            last_timestamp=last_timestamp,
            periods=periods
        )

        # --- Save Constant Results ---
        try:
            pred_dir = "integrated_predictions"
            os.makedirs(pred_dir, exist_ok=True)
            pred_path = os.path.join(pred_dir, f"{item_id}.json")
            predictions_data = {
                "item_id": item_id,
                "current_buy_price": last_buy_price,
                "current_sell_price": last_sell_price,
                "current_timestamp": last_timestamp.isoformat(),
                "best_model_type": "Constant (Naive)",
                "models": {
                    "Constant": {
                        "method": "Naive (Last Price Repeated for Buy/Sell)",
                        "performance": {"note": "Prices are constant, no training metrics"},
                        "future_predictions": format_predictions_for_saving(const_buy_preds, const_sell_preds)
                    }
                }
            }
            with open(pred_path, 'w') as f:
                json.dump(predictions_data, f, indent=2, cls=NpEncoder)
            logging.info(f"Saved constant model results for {item_id} to {pred_path}")

            if graph:
                # --- Plot Constant Results ---
                plot_predictions_combined(
                    df=item_df,
                    arima_buy_preds=const_buy_preds, arima_sell_preds=const_sell_preds,
                    lstm_buy_preds=[], lstm_sell_preds=[],
                    ensemble_buy_preds=[], ensemble_sell_preds=[],
                    vecm_buy_preds=[], vecm_sell_preds=[],
                    item_id=f"{item_id}_constant"
                )
                logging.info(f"Saved constant model plot for {item_id}.")
        except Exception as e_save_const:
            logging.error(f"Error saving/plotting constant model results for {item_id}: {e_save_const}")

        # --- Print Constant Predictions ---
        if const_buy_preds and const_sell_preds:
            print(f"\nPredicted prices for {item_id} (Constant Model):")
            print(f"  {'Time':<20} | {'Buy Price':<12} | {'Sell Price':<12}")
            print("-" * 40)
            for buy_pred, sell_pred in zip(const_buy_preds, const_sell_preds):
                pred_time_str = buy_pred['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                buy_price_str = f"{buy_pred['predicted_price']:.2f}"
                sell_price_str = f"{sell_pred['predicted_price']:.2f}"
                if TERMCOLOR_AVAILABLE:
                    colored_buy = colored(buy_price_str, 'green')
                    colored_sell = colored(sell_price_str, 'red')
                else:
                    colored_buy = buy_price_str
                    colored_sell = sell_price_str
                print(f"  {pred_time_str} | {colored_buy:>12} | {colored_sell:>12}")
        logging.info(f"Finished processing item: {item_id} (Constant)")
        return
    # --- End Constant Check ---

    # --- 2. Run All Models (VECM, ARIMA, LSTM) ---
    logging.info(f"Item {item_id} has varying prices. Running all models.")

    # --- Initialize model results ---
    vecm_success, vecm_buy_preds, vecm_sell_preds, vecm_metrics, vecm_msg, vecm_last_buy, vecm_last_sell, vecm_last_ts = False, None, None, None, "", None, None, None
    arima_success, arima_buy_preds, arima_sell_preds, arima_info, arima_msg, arima_last_buy, arima_last_sell, arima_last_ts = False, None, None, None, "", None, None, None
    lstm_success, lstm_buy_preds, lstm_sell_preds, lstm_metrics, lstm_msg, lstm_last_buy, lstm_last_sell, lstm_last_ts = False, None, None, None, "", None, None, None
    ensemble_buy_preds, ensemble_sell_preds = None, None
    last_buy_price, last_sell_price = None, None
    last_timestamp = None
    # --- End Initialization ---

    # --- Run VECM ---
    vecm_success, vecm_buy_preds, vecm_sell_preds, vecm_metrics, vecm_msg, vecm_last_buy, vecm_last_sell, vecm_last_ts = \
        process_item_vecm(df, item_id, periods=periods, max_lag=2, max_coint_rank=2, train_split_ratio=0.8)
    if vecm_success:
        logging.info(f"VECM successful for {item_id}.")
        if last_buy_price is None and vecm_last_buy is not None:
            last_buy_price = vecm_last_buy
            last_sell_price = vecm_last_sell
            last_timestamp = vecm_last_ts
    else:
        logging.info(f"VECM not suitable or failed for {item_id} ({vecm_msg}).")
    # --- End VECM ---

    # --- Run ARIMA ---
    arima_success, arima_buy_preds, arima_sell_preds, arima_info, arima_msg, arima_last_buy, arima_last_sell, arima_last_ts = \
        process_item_arima_primary(df, item_id, periods=periods, train_split_ratio=0.8)
    if arima_success:
        logging.info(f"ARIMA successful for {item_id}.")
        if last_buy_price is None and arima_last_buy is not None:
            last_buy_price = arima_last_buy
            last_sell_price = arima_last_sell
            last_timestamp = arima_last_ts
    else:
        logging.info(f"ARIMA failed for {item_id} ({arima_msg}).")
    # --- End ARIMA ---

    # --- Run LSTM ---
    lstm_success, lstm_buy_preds, lstm_sell_preds, lstm_metrics, lstm_msg, lstm_last_buy, lstm_last_sell, lstm_last_ts = \
        process_item_lstm_primary(df, item_id, periods=periods, seq_length=24, epochs=50)
    if lstm_success:
        logging.info(f"LSTM successful for {item_id}.")
        if last_buy_price is None and lstm_last_buy is not None:
            last_buy_price = lstm_last_buy
            last_sell_price = lstm_last_sell
            last_timestamp = lstm_last_ts
    else:
        logging.info(f"LSTM failed for {item_id} ({lstm_msg}).")
    # --- End LSTM ---

    # --- Combine Predictions (Buy and Sell separately) ---
    if (arima_success and lstm_success and
        arima_buy_preds and lstm_buy_preds and len(arima_buy_preds) == len(lstm_buy_preds) and
        arima_sell_preds and lstm_sell_preds and len(arima_sell_preds) == len(lstm_sell_preds)):

        ensemble_buy_preds = combine_predictions_simple_average_single(arima_buy_preds, lstm_buy_preds, item_id, "buy")
        ensemble_sell_preds = combine_predictions_simple_average_single(arima_sell_preds, lstm_sell_preds, item_id, "sell")

        if ensemble_buy_preds and ensemble_sell_preds:
            logging.info(f"Created ARIMA-LSTM ensemble for {item_id} (buy and sell).")
        else:
            logging.error(f"Failed to create ARIMA-LSTM ensemble for {item_id} (buy or sell).")
            ensemble_buy_preds, ensemble_sell_preds = None, None
    else:
        logging.info(f"Could not create ARIMA-LSTM ensemble for {item_id}.")
        ensemble_buy_preds, ensemble_sell_preds = None, None
    # --- End Combine Predictions ---

    # --- Final Check for Last Prices/Timestamp ---
    if last_buy_price is None or last_timestamp is None:
        logging.error(f"Could not determine last prices/timestamp for {item_id}. Cannot finalize results.")
        return
    if last_sell_price is None:
        logging.warning(f"Could not determine last sell price for {item_id}. Using last buy price as fallback.")
        last_sell_price = last_buy_price
    # --- End Final Check ---

    # --- 3. Determine Best Model Based on In-Sample MAE (for buy price) ---
    model_mae_info = {}
    if vecm_success and vecm_metrics and not np.isnan(vecm_metrics.get("insample_mae")):
        model_mae_info["VECM"] = vecm_metrics["insample_mae"]
    if arima_success and arima_info and not np.isnan(arima_info.get("mae")):
        model_mae_info["ARIMA"] = arima_info["mae"]
    if lstm_success and lstm_metrics and not np.isnan(lstm_metrics.get("mae")):
        model_mae_info["LSTM"] = lstm_metrics["mae"]

    best_model_type = "None"
    best_buy_predictions = []
    best_sell_predictions = []
    best_mae = None

    if model_mae_info:
        best_model_type = min(model_mae_info, key=model_mae_info.get)
        best_mae = model_mae_info[best_model_type]
        logging.info(f"Model In-Sample MAEs (buy price) for {item_id}: {model_mae_info}")
        logging.info(f"Selected best model for {item_id} based on lowest In-Sample MAE (buy): {best_model_type} (MAE={best_mae:.4f})")

        # Assign the corresponding predictions (buy and sell)
        model_pred_map = {
            "VECM": (vecm_buy_preds, vecm_sell_preds),
            "ARIMA": (arima_buy_preds, arima_sell_preds),
            "LSTM": (lstm_buy_preds, lstm_sell_preds),
        }
        if best_model_type in model_pred_map:
            best_buy_predictions, best_sell_predictions = model_pred_map[best_model_type]

        if not best_buy_predictions:
            logging.warning(f"Selected best model {best_model_type} has no buy predictions for {item_id}.")

    else:
        logging.warning(f"No valid In-Sample MAE found for any successful model for {item_id}.")
    # --- End Determine Best Model ---

    # --- 4. Save Combined Results and Plot ---
    try:
        pred_dir = "integrated_predictions"
        os.makedirs(pred_dir, exist_ok=True)
        pred_path = os.path.join(pred_dir, f"{item_id}.json")

        predictions_data = {
            "item_id": item_id,
            "current_buy_price": last_buy_price,
            "current_sell_price": last_sell_price,
            "current_timestamp": last_timestamp.isoformat() if last_timestamp else None,
            "best_model_type": best_model_type,
            "best_model_in_sample_mae_buy": best_mae,
            "models": {}
        }

        # Add VECM results
        if vecm_success:
            vecm_info_for_save = vecm_metrics if vecm_metrics else {}
            # Convert numpy types to Python native types
            for key, value in vecm_info_for_save.items():
                if isinstance(value, (np.integer, np.floating, np.bool_)):
                    vecm_info_for_save[key] = value.item() if hasattr(value, 'item') else float(value)
                elif isinstance(value, np.ndarray):
                    vecm_info_for_save[key] = value.tolist()
            if "insample_mae" in vecm_info_for_save and not np.isnan(vecm_info_for_save["insample_mae"]):
                vecm_info_for_save["insample_mae"] = float(vecm_info_for_save["insample_mae"])
            predictions_data["models"]["VECM"] = {
                "info": vecm_info_for_save,
                "future_predictions": format_predictions_for_saving(vecm_buy_preds, vecm_sell_preds)
            }
        # Add ARIMA results
        if arima_success:
            arima_info_for_save = arima_info if arima_info else {}
            # Convert numpy types to Python native types
            for key, value in arima_info_for_save.items():
                if isinstance(value, (np.integer, np.floating, np.bool_)):
                    arima_info_for_save[key] = value.item() if hasattr(value, 'item') else float(value)
                elif isinstance(value, np.ndarray):
                    arima_info_for_save[key] = value.tolist()
            if "mae" in arima_info_for_save and not np.isnan(arima_info_for_save["mae"]):
                arima_info_for_save["mae"] = float(arima_info_for_save["mae"])
            predictions_data["models"]["ARIMA"] = {
                "info": arima_info_for_save,
                "future_predictions": format_predictions_for_saving(arima_buy_preds, arima_sell_preds)
            }
        # Add LSTM results
        if lstm_success:
            lstm_metrics_for_save = lstm_metrics if lstm_metrics else {}
            # Convert numpy types to Python native types
            for key, value in lstm_metrics_for_save.items():
                if isinstance(value, (np.integer, np.floating, np.bool_)):
                    lstm_metrics_for_save[key] = value.item() if hasattr(value, 'item') else float(value)
                elif isinstance(value, np.ndarray):
                    lstm_metrics_for_save[key] = value.tolist()
            if "mae" in lstm_metrics_for_save and not np.isnan(lstm_metrics_for_save["mae"]):
                lstm_metrics_for_save["mae"] = float(lstm_metrics_for_save["mae"])
            predictions_data["models"]["LSTM"] = {
                "performance": lstm_metrics_for_save,
                "future_predictions": format_predictions_for_saving(lstm_buy_preds, lstm_sell_preds)
            }
        # Add Ensemble results
        if ensemble_buy_preds and ensemble_sell_preds:
            predictions_data["models"]["Ensemble"] = {
                "method": "Simple Average (ARIMA+LSTM)",
                "future_predictions": format_predictions_for_saving(ensemble_buy_preds, ensemble_sell_preds)
            }

        with open(pred_path, 'w') as f:
            json.dump(predictions_data, f, indent=2, cls=NpEncoder)
        logging.info(f"Saved combined results for {item_id} to {pred_path}")

        if graph:
            # --- Plotting ---
            plot_predictions_combined(
                df=item_df,
                arima_buy_preds=arima_buy_preds if arima_success else [],
                arima_sell_preds=arima_sell_preds if arima_success else [],
                lstm_buy_preds=lstm_buy_preds if lstm_success else [],
                lstm_sell_preds=lstm_sell_preds if lstm_success else [],
                ensemble_buy_preds=ensemble_buy_preds if ensemble_buy_preds else [],
                ensemble_sell_preds=ensemble_sell_preds if ensemble_sell_preds else [],
                vecm_buy_preds=vecm_buy_preds if vecm_success else [],
                vecm_sell_preds=vecm_sell_preds if vecm_success else [],
                item_id=item_id
            )
            logging.info(f"Saved combined plot for {item_id}.")
            # --- End Plotting ---
        logging.info(f"Finished processing item: {item_id}")
    except Exception as e:
        logging.error(f"Error saving/plotting final combined results for {item_id}: {e}")
        logging.debug(traceback.format_exc())
        return
    # --- End Save Results and Plot ---

    # --- 5. Print Predicted Prices to Console ---
    if best_buy_predictions and best_model_type != "None":
        print(f"\nPredicted prices for {item_id} (Best Model: {best_model_type}, In-Sample MAE (buy): {best_mae:.4f}):")
        print(f"  {'Time':<20} | {'Buy Price':<12} | {'Sell Price':<12}")
        print("-" * 40)
        sell_preds_to_use = best_sell_predictions if best_sell_predictions else [{}] * len(best_buy_predictions)
        for buy_pred, sell_pred in zip(best_buy_predictions, sell_preds_to_use):
            pred_time_str = buy_pred['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            buy_price_str = f"{buy_pred['predicted_price']:.2f}" if not np.isnan(buy_pred['predicted_price']) else "N/A"
            sell_price_val = sell_pred.get('predicted_price', None)
            if sell_price_val is not None and not np.isnan(sell_price_val):
                sell_price_str = f"{sell_price_val:.2f}"
            else:
                sell_price_str = "N/A"

            if TERMCOLOR_AVAILABLE:
                colored_buy = colored(buy_price_str, 'green') if buy_price_str != "N/A" else buy_price_str
                colored_sell = colored(sell_price_str, 'red') if sell_price_str != "N/A" else sell_price_str
            else:
                colored_buy = buy_price_str
                colored_sell = sell_price_str
            print(f"  {pred_time_str} | {colored_buy:>12} | {colored_sell:>12}")
    else:
        print(f"\nNo predictions available for {item_id} or no model selected.")
    # --- End Print Predicted Prices ---

# --- Command-Line Interface (CLI) Entry Point ---
def main_cli_full():
    parser = argparse.ArgumentParser(description="Predict Hypixel Bazaar buy/sell prices using integrated models (VECM, ARIMA, LSTM). Selection based on In-Sample MAE (buy).")
    parser.add_argument("item_id", help="The item ID to predict (e.g., ENCHANTED_DIAMOND).")
    parser.add_argument("--data-file", default="bazaar_data.jsonl", help="Path to the input NDJSON data file (default: bazaar_data.jsonl).")
    parser.add_argument("--periods", type=int, default=24, help="Number of future periods (hours) to predict (default: 24).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--graph", "-g", action="store_true", help="Generate and save plots")
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.periods <= 0:
        logging.error(f"Invalid number of periods: {args.periods}. Must be a positive integer.")
        return
    try:
        if not os.path.exists(args.data_file):
            logging.error(f"Data file '{args.data_file}' not found.")
            return
        df = load_data(args.data_file)
        process_single_item_full(df, args.item_id, periods=args.periods, graph=args.graph)
        logging.info(f"Integrated Model CLI Processing complete for item: {args.item_id} for {args.periods} periods!")
        logging.info(f"Results saved to integrated_predictions")
        logging.info(f"Models saved to integrated_models")
        logging.info(f"Plots saved to integrated_plots")
    except Exception as e:
        logging.error(f"Unexpected error in main: {e}")
        logging.debug(traceback.format_exc())

# --- Script Execution Entry Point ---
if __name__ == "__main__":
    main_cli_full()