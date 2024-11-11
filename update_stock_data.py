import os
import requests
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Debug
print("FIREBASE_PROJECT_ID:", os.getenv("FIREBASE_PROJECT_ID"))
print("FIREBASE_CLIENT_EMAIL:", os.getenv("FIREBASE_CLIENT_EMAIL"))
print("FIREBASE_PRIVATE_KEY exists:", os.getenv("FIREBASE_PRIVATE_KEY"))


# Access Firebase credentials from environment variables
if not firebase_admin._apps:
    cred = credentials.Certificate({
        "type": os.getenv("FIREBASE_TYPE"),
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY"), 
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
        "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
        "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_CERT_URL"),
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL")
    })
    firebase_admin.initialize_app(cred)

db = firestore.client()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# Function to fetch and update stock data in Firestore
# Function to fetch and update stock data in Firestore
def update_stock_data(symbol, api_key):
    """
    Updates stock data in Firebase Firestore.
    Only adds new entries and does not overwrite existing ones.
    """
    # Firestore reference for the collection of each stock
    stock_collection_ref = db.collection("stock_data").document(symbol).collection("daily_data")

    # Fetch current data from Firestore
    existing_dates = set()
    docs = stock_collection_ref.stream()
    for doc in docs:
        existing_dates.add(doc.id)  # Store document IDs (dates) that are already in Firestore

    # Fetch from Twelve Data API
    url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api_key}'
    response = requests.get(url)
    data = response.json()

    if "values" in data:
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

        # Iterate through the data and write each row as a separate document
        for _, row in df.iterrows():
            date_str = row['Date'].strftime("%Y-%m-%d")
            if date_str not in existing_dates:  # Only add new data
                doc_ref = stock_collection_ref.document(date_str)
                row_data = row.drop('Date').dropna().to_dict()
                doc_ref.set(row_data)

# Update the list of stocks
stocks = {
    "NVIDIA": "NVDA",
    "Advanced Micro Devices": "AMD",
    "Micron Technology": "MU",
    "Astera Labs Inc": "ALAB",
    "Arm": "ARM",
    "Alphabet 1": "GOOGL",
    "Alphabet 2": "GOOG",
    "Broadcom Inc.": "AVGO",
    "Amazon": "AMZN",
    "NXP Semiconductors": "NXPI",
    "Microsoft": "MSFT",
    "TSMC": "2330",
    "SK Hynix": "000660",
    "Meta Platforms": "META",
    "Palantir Technologies": "PLTR",
    "Marvell Technology Inc": "MRVL",
    "CrowdStrike": "CRWD",
    "Arista": "ANED"
}

# Update each stock's data
for stock_name, stock_symbol in stocks.items():
    update_stock_data(stock_symbol, TWELVE_DATA_API_KEY)
