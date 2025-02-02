import os
import requests
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import json
import time
from datetime import datetime, time as dt_time
import pytz  # if you need timezone conversions

# --- Load environment variables from .env file To run locally ---
#region
#from dotenv import load_dotenv  # To run locally

# Load environment variables from .env file
#load_dotenv()  # To run locally

# Debug
#print("FIREBASE_PROJECT_ID:", os.getenv("FIREBASE_PROJECT_ID"))
#print("FIREBASE_CLIENT_EMAIL:", os.getenv("FIREBASE_CLIENT_EMAIL"))
#print("FIREBASE_PRIVATE_KEY exists:", os.getenv("FIREBASE_PRIVATE_KEY"))
#endregion


# Define market windows (these are just examples)
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)
PRE_MARKET_START = dt_time(7, 0)
AFTER_HOURS_END = dt_time(20, 0)

# Get current EST time (assuming the server is not in EST)
est = pytz.timezone('US/Eastern')
now_est = datetime.now(est).time()

# Check if current time is within any of the desired update windows.
if not ((PRE_MARKET_START <= now_est <= AFTER_HOURS_END)):
    print("Market is closed. Skipping update for now.")
    exit(0)

# Access Firebase credentials from environment variables
if not firebase_admin._apps:
    cred = credentials.Certificate({
        "type": "service_account",
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY"), 
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL")
    })
    firebase_admin.initialize_app(cred)

db = firestore.client()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# Function to fetch and update stock data in Firestore
# Function to fetch and store stock data as JSON in Firestore
def update_stock_data(symbol, api_key):
    """
    Updates stock data in Firebase Firestore.
    Fetches data from Twelve Data API and stores it as a JSON document in Firestore.
    """
    # Firestore reference for the JSON document of each stock
    stock_doc_ref = db.collection("stock_data_json").document(symbol)

    # Fetch from Twelve Data API
    url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api_key}'
    response = requests.get(url)
    data = response.json()

    if "values" in data:

        #debugging
        #for i, item in enumerate(data["values"]):
            #if i >= 5:
                #break
            #print(item)

        values = data["values"]
        df = pd.DataFrame(values)
        df = df.rename(columns={
            "datetime": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })

        # Ensure all data is in the correct format
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by="Date")
        df = df.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name != 'Date' else x)
        df["Date"] = df["Date"].dt.strftime('%Y-%m-%d')  # Convert Date and time back to string format

        # Convert DataFrame to JSON format
        data_json = df.to_json(orient="records")

        # Get current Eastern Time
        est = pytz.timezone('US/Eastern')
        current_time = datetime.now(est).strftime('%Y-%m-%d %H:%M')

        # Store JSON data in Firestore
        stock_doc_ref.set({
            "data": data_json,
            "last_updated": current_time # Store the current time of update
            })
    else:
        print(f"Error retrieving data for {symbol}. Response: {data}")

# Update the list of stocks
stocks = {
    "NVIDIA": "NVDA",
    #"Advanced Micro Devices": "AMD",
    "Micron Technology": "MU",
    "Astera Labs Inc": "ALAB",
    "Arm": "ARM",
    "Alphabet 1": "GOOGL",
    "Broadcom Inc.": "AVGO",
    "NXP Semiconductors": "NXPI",
    "Palantir Technologies": "PLTR",
    "Marvell Technology Inc": "MRVL",
    "Palo Alto Networks": "PANW",
    "Gartner": "IT",
    "Oracle": "ORCL",
    "Service now": "NOW",
    #"Applied Materials": "AMAT",
    "Arista": "ANET",
    "Cadence Design Systems": "CDNS",
    "Progress Software": "PRGS",
    "Synopsys": "SNPS",
    "AMAZON": "AMZN"
}

# Update each stock's data
for stock_name, stock_symbol in stocks.items():
    update_stock_data(stock_symbol, TWELVE_DATA_API_KEY)
    print(f"updating {stock_symbol}...")
    time.sleep(8)  # Delay to avoid exceeding API rate limit

