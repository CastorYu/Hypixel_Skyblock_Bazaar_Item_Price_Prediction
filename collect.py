import requests
import time
import json
import os
from datetime import datetime

# Configuration
API_KEY = input("API_key: ")  # Replace with your actual Hypixel API key
OUTPUT_FILE = "bazaar_data.jsonl"
SAVE_INTERVAL = 20  # seconds

def fetch_bazaar_data():
    """Fetches current bazaar data from Hypixel API"""
    url = f"https://api.hypixel.net/skyblock/bazaar?key={API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code} - {response.text}")
        return None

def save_to_jsonl(data):
    """Saves bazaar data to JSONL file (one JSON object per line)"""
    if not data or not data.get("success", False):
        return False
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(OUTPUT_FILE, 'a') as f:
        for product_id, product_data in data["products"].items():
            quick_status = product_data["quick_status"]
            
            # Create a single-line JSON object for this product
            record = {
                "timestamp_id": timestamp_id,
                "timestamp": timestamp,
                "item_id": product_id,
                "name": product_data.get("name", product_id),
                "buy_price": quick_status["buyPrice"],
                "sell_price": quick_status["sellPrice"],
                "buy_volume": quick_status["buyVolume"],
                "sell_volume": quick_status["sellVolume"],
                "buy_moving_week": quick_status["buyMovingWeek"],
                "sell_moving_week": quick_status["sellMovingWeek"],
                "buy_orders": quick_status["buyOrders"],
                "sell_orders": quick_status["sellOrders"]
            }
            
            # Write as a single line (JSONL requirement)
            f.write(json.dumps(record) + '\n')
    
    print(f"Appended {len(data['products'])} records to {OUTPUT_FILE}")
    return True

def main():
    print("Starting SkyBlock Bazaar Price Tracker (JSONL format)...")
    print(f"Data will be saved to {OUTPUT_FILE} every {SAVE_INTERVAL} seconds")
    print("Press Ctrl+C to stop")
    
    # Create file with proper headers if it doesn't exist
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w') as f:
            pass  # Create empty file
    
    try:
        while True:
            bazaar_data = fetch_bazaar_data()
            if bazaar_data:
                save_to_jsonl(bazaar_data)
            
            # Wait for the next interval
            time.sleep(SAVE_INTERVAL)
            
    except KeyboardInterrupt:
        print("\nStopping SkyBlock Bazaar Price Tracker...")

if __name__ == "__main__":
    main()