# utils/datetime_utils.py
"""
Module for robust datetime conversion utilities.
"""

import numpy as np
import pandas as pd
import polars as pl
import logging
from datetime import timezone, datetime

# --- Helper Function: Datetime Conversion ---
def convert_numpy_datetime64_to_python_datetime(np_dt64):
    """
    Converts a numpy.datetime64 object to a timezone-aware Python datetime object (UTC).
    Uses Pandas as primary method (more reliable with numpy datetime64), 
    with a fallback to Polars if Pandas fails.
    """
    if not isinstance(np_dt64, np.datetime64):
        raise TypeError(f"Input must be a numpy.datetime64 object, got {type(np_dt64)}")
    
    try:
        # --- Attempt 1: Using Pandas (Most Reliable for numpy datetime64) ---
        # pd.to_datetime handles various numpy datetime64 dtypes well [[6]]
        converted_datetime = pd.to_datetime(np_dt64).to_pydatetime()
        # Ensure it's timezone-aware in UTC
        if converted_datetime.tzinfo is None:
            converted_datetime = converted_datetime.replace(tzinfo=timezone.utc)
        return converted_datetime
    except Exception as e_pd:
        logging.warning(f"Pandas conversion failed for {np_dt64}: {e_pd}. Trying Polars fallback.")
        try:
            # --- Attempt 2: Fallback using Polars ---
            # Convert numpy.datetime64 to Unix timestamp, then to Python datetime
            # This avoids direct conversion issues with Polars [[2]]
            timestamp = np_dt64.astype('datetime64[ns]').astype(np.int64) / 1e9
            converted_datetime = datetime.utcfromtimestamp(timestamp).replace(tzinfo=timezone.utc)
            return converted_datetime
        except Exception as e_pl:
            # --- Final Error ---
            error_msg = f"Error converting datetime64 {np_dt64}: Pandas error: {e_pd}, Polars error: {e_pl}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e_pl  # Chain the last exception
        