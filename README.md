# Hypixel Skyblock Bazaar Item Price Prediction

This project implements a machine learning-based prediction system for item prices in the Hypixel Skyblock Bazaar. It uses multiple prediction models including VECM (Vector Error Correction Model), ARIMA (Autoregressive Integrated Moving Average), and LSTM (Long Short-Term Memory) to forecast both buy and sell prices of items.

## Features

- Multiple prediction models:
  - VECM (Vector Error Correction Model)
  - ARIMA (Autoregressive Integrated Moving Average)
  - LSTM (Long Short-Term Memory Neural Network)
  - Ensemble model combining ARIMA and LSTM predictions
  - Constant/Naive model for items with stable prices
- Automated model selection based on in-sample MAE (Mean Absolute Error)
- Price predictions for both buy and sell prices
- Visualization of predictions through plots
- Comprehensive logging system
- Command-line interface for easy use

## Requirements

- Python 3.x
- Required packages (install via `pip install -r requirements.txt`):
  - polars
  - numpy
  - requests
  - scikit-learn
  - statsmodels
  - tensorflow
  - matplotlib
  - joblib
  - python-dateutil

## Project Structure

```
Skyblock_Price_Prediction/
├── cli.py               # Price prediction CLI tool
├── collect.py           # Data collection script
├── README.md            # Project documentation (this file)
├── requirements.txt     # Python dependencies
├── integrated_models/   # Saved machine learning models
├── integrated_plots/    # Generated price prediction plots
└── integrated_predictions/  # JSON prediction results
```

## Usage

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the prediction script for a specific item:
   ```bash
   python main.py ITEM_ID [--data-file DATA_FILE] [--periods PERIODS] [--debug] [--graph]
   ```

   Arguments:
   - `ITEM_ID`: The ID of the item to predict (e.g., ENCHANTED_DIAMOND)
   - `--data-file`: Path to the input NDJSON data file (default: bazaar_data.jsonl)
   - `--periods`: Number of future periods (hours) to predict (default: 24)
   - `--debug`: Enable debug logging
   - `--graph`: Generate and save plots

## Output

The script generates several outputs:

1. Predictions are saved in the `integrated_predictions` directory as JSON files
2. Model files are saved in the `integrated_models` directory
3. Plots are saved in the `integrated_plots` directory (when --graph is enabled)
4. Console output showing predicted prices with color-coded buy/sell values
5. Detailed logs in `cli.log`

## Example Output Format

```
Predicted prices for ITEM_ID (Best Model: ARIMA, In-Sample MAE (buy): 0.1234):
  Time                 | Buy Price    | Sell Price  
----------------------------------------
  2024-01-01 00:00:00 |      100.50  |       98.25
  2024-01-01 01:00:00 |      101.75  |       99.00
  ...
```

## Notes

- The system automatically selects the best model based on the lowest in-sample MAE for buy prices
- For items with constant/stable prices, a simple naive prediction model is used
- All predictions include both buy and sell prices when available
- The project uses colored console output when available (through termcolor package)

## Contributing

Feel free to submit issues and enhancement requests!
