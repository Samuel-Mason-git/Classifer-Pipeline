# crypto-classifier\src\get_data.py
from pathlib import Path
import os
from utils.env import load_environment
from utils.api import call_api
import pandas as pd

# -----------------------
# Load environment
# -----------------------
load_environment()
if not os.getenv("COINGECKO_API_KEY"):
    raise ValueError("COINGECKO_API_KEY missing; .env not loaded")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

# -----------------------
# API config
# -----------------------
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
COINGECKO_HEADERS = {
    "accept": "application/json",
    "x-cg-demo-api-key": COINGECKO_API_KEY,
}

COINGECKO_PARAMS = {
    "vs_currency": "usd",
    "days": 365,
    "interval": "daily",
}

# -----------------------
# Fetch
# -----------------------
def fetch_data(url, headers, params, debug=False):
    return call_api(url, headers=headers, params=params, debug=debug)

# -----------------------
# Processing
# -----------------------
def process_data(data):
    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    market_caps = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])

    df = prices.merge(market_caps, on="timestamp").merge(volumes, on="timestamp")
    df["date_time"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df

# -----------------------
# Processing
# -----------------------
def save_data(df):
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "raw" / "btc_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved raw data to {output_path}")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    raw = fetch_data(url=COINGECKO_URL, headers=COINGECKO_HEADERS, params=COINGECKO_PARAMS)
    df = process_data(raw)
    save_data(df)