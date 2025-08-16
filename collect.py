import requests
import time
import json
import os
from datetime import datetime

# Configuration
#API_KEY = input("API_key: ")   Replace with your actual Hypixel API key
OUTPUT_FILE = "bazaar_data.jsonl"
SAVE_INTERVAL = 20  # seconds

def fetch_bazaar_data():
    """Fetches current bazaar data from Hypixel API"""
    url = f"https://api.hypixel.net/skyblock/bazaar"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code} - {response.text}")
        return None

def _safe_top_buy_price(buy_summary):
    """
    Returns the highest buy order price (best buy) from buy_summary.
    If summary is missing/empty or malformed, returns None.
    """
    try:
        if not buy_summary:
            return None
        prices = [entry.get("pricePerUnit") for entry in buy_summary if "pricePerUnit" in entry]
        return max(prices) if prices else None
    except Exception:
        return None

def _safe_top_sell_price(sell_summary):
    """
    Returns the lowest sell order price (best sell) from sell_summary.
    If summary is missing/empty or malformed, returns None.
    """
    try:
        if not sell_summary:
            return None
        prices = [entry.get("pricePerUnit") for entry in sell_summary if "pricePerUnit" in entry]
        return min(prices) if prices else None
    except Exception:
        return None

def save_to_jsonl(data):
    """Saves bazaar data to JSONL file (one JSON object per line)"""
    if not data or not data.get("success", False):
        return False

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(OUTPUT_FILE, 'a') as f:
        for product_id, product_data in data["products"].items():
            quick_status = product_data.get("quick_status", {})
            buy_summary = product_data.get("buy_summary", [])
            sell_summary = product_data.get("sell_summary", [])

            top_buy_order_price = _safe_top_buy_price(buy_summary)
            top_sell_order_price = _safe_top_sell_price(sell_summary)

            record = {
                "timestamp_id": timestamp_id,
                "timestamp": timestamp,
                "item_id": product_id,
                "name": product_data.get("name", product_id),

                # actual buy and sell prices
                "buy_price": top_buy_order_price,
                "sell_price": top_sell_order_price,
                "buy_volume": quick_status.get("buyVolume"),
                "sell_volume": quick_status.get("sellVolume"),
                "buy_moving_week": quick_status.get("buyMovingWeek"),
                "sell_moving_week": quick_status.get("sellMovingWeek"),
                "buy_orders": quick_status.get("buyOrders"),
                "sell_orders": quick_status.get("sellOrders"),
            }

            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Appended {len(data['products'])} records to {OUTPUT_FILE}")
    return True

def main():
    print("Starting SkyBlock Bazaar Price Tracker (JSONL format)...")
    print(f"Data will be saved to {OUTPUT_FILE} every {SAVE_INTERVAL} seconds")
    print("Press Ctrl+C to stop")

    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            pass  # Create empty file

    try:
        while True:
            bazaar_data = fetch_bazaar_data()
            if bazaar_data:
                save_to_jsonl(bazaar_data)
            time.sleep(SAVE_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopping SkyBlock Bazaar Price Tracker...")

if __name__ == "__main__":
    main()