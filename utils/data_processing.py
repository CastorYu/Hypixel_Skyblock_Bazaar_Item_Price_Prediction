# utils/data_processing.py
"""
Module for data loading and preprocessing utilities.
"""

import polars as pl
import pandas as pd
import numpy as np
import logging

# --- Data Loading Function ---
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
        logging.error(f"Error loading {file_path}: {str(e)}")
        raise

# --- Constant Series Detection ---
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
