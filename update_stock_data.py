import requests
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import streamlit as st

# Access Firebase credentials from Streamlit secrets
if not firebase_admin._apps:
    cred = credentials.Certificate({
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": st.secrets["firebase"]["private_key"],
        "client_email": st.secrets["firebase"]["client_email"],
        "client_id": st.secrets["firebase"]["client_id"],
        "auth_uri": st.secrets["firebase"]["auth_uri"],
        "token_uri": st.secrets["firebase"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
    })
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Access the API key from the secrets.toml file
TWELVE_DATA_API_KEY = st.secrets["twelve_data"]["api_key"]

# Function to fetch and update stock data in Firestore
def update_stock_data(symbol, api_key):
    """
    Updates stock data in Firebase Firestore.
    Only adds new entries and does not overwrite existing ones.
    """
    # Firestore reference
    doc_ref = db.collection("stock_data").document(symbol)

    # Fetch current data from Firestore
    doc = doc_ref.get()
    if doc.exists:
        existing_data = pd.DataFrame(doc.to_dict())
        last_date = pd.to_datetime(existing_data.index).max()
    else:
        existing_data = pd.DataFrame()
        last_date = None

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

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by="Date")
        df = df.apply(lambda x: pd.to_numeric(x, errors='ignore') if x.name != 'Date' else x)
        df.set_index("Date", inplace=True)

        # Filter new data
        if last_date:
            new_data = df[df.index > last_date]
        else:
            new_data = df

        # Update Firestore if there is new data
        if not new_data.empty:
            updated_data = pd.concat([existing_data, new_data])
            doc_ref.set(updated_data.to_dict())

# Update the list of stocks
stocks = {
    "NVIDIA": "NVDA"
}

# Update each stock's data
for stock_name, stock_symbol in stocks.items():
    update_stock_data(stock_symbol, TWELVE_DATA_API_KEY)
