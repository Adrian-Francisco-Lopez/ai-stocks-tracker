import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials, firestore

#### Initializing variables ####
#region

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

# Initialize session state
if "stock_data" not in st.session_state:
    st.session_state.stock_data = {}

# Dictionary of stocks
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

#endregion

#### Functions ####
#region

# Fetch stock data from Firebase Firestore
def get_stock_data_from_firebase(symbol):
    """
    This function fetches stock data from Firebase Firestore.
    """
    # Firestore reference for the collection of each stock
    stock_collection_ref = db.collection("stock_data").document(symbol).collection("daily_data")

    # Fetch all documents from the collection
    docs = stock_collection_ref.stream()
    data = []
    for doc in docs:
        doc_data = doc.to_dict()
        doc_data["Date"] = pd.to_datetime(doc.id)  # Use the document ID (date) as the "Date" field
        data.append(doc_data)

    # Convert to DataFrame
    if data:
        df = pd.DataFrame(data)
        df.set_index("Date", inplace=True)
        df = df.sort_index()
        return df
    else:
        return pd.DataFrame()

# Fetch stock data from Twelve Data API
old_function = """def get_stock_data(symbol, api_key):
    # Firestore reference
    doc_ref = db.collection("stock_data").document(symbol)

    # Check if data already exists in Firestore
    doc = doc_ref.get()
    if doc.exists:
        return pd.DataFrame(doc.to_dict())

    # Fetch from Twelve Data API if not available
    url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api_key}'
    response = requests.get(url)
    data = response.json()

    # Convert data to DataFrame and store in Firestore
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

        # Store in Firestore
        doc_ref.set(df.to_dict())
        return df
    else:
        st.error("Error retrieving data. Please try again later.")
        return pd.DataFrame()"""

#endregion

#### Main app logic ####
#region

# Title of the webpage
st.set_page_config(page_title="AI-related Stock Tracker")
st.title("AI-related Stock Tracker")

# Loop through all stocks
for stock_name, stock_symbol in stocks.items():
    # Fetch stock data from Firebase
    if stock_symbol not in st.session_state.stock_data:
        st.session_state.stock_data[stock_symbol] = get_stock_data_from_firebase(stock_symbol)

    # Display stock data
    stock_data = st.session_state.stock_data[stock_symbol]

    # Check if data is available before proceeding
    if not stock_data.empty:
        with st.container():
            st.write(f"### {stock_name} ({stock_symbol}) Stock Price")
            
            # User selection for time range
            time_range = st.selectbox(
                f"Select Time Range for {stock_name}:",
                ["Full Range", "Last 5 Years", "Last 3 Years", "Last Year", "Last 6 Months", "Last Month", "Last Week"],
                key=f"{stock_symbol}_time_range"
            )

            # Filtering data based on user selection
            today = datetime.now()
            if time_range == "Full Range":
                filtered_data = stock_data
            elif time_range == "Last 5 Years":
                filtered_data = stock_data[stock_data.index >= (today - pd.DateOffset(years=5))]
            elif time_range == "Last 3 Years":
                filtered_data = stock_data[stock_data.index >= (today - pd.DateOffset(years=3))]
            elif time_range == "Last Year":
                filtered_data = stock_data[stock_data.index >= (today - pd.DateOffset(years=1))]
            elif time_range == "Last 6 Months":
                filtered_data = stock_data[stock_data.index >= (today - pd.DateOffset(months=6))]
            elif time_range == "Last Month":
                filtered_data = stock_data[stock_data.index >= (today - pd.DateOffset(months=1))]
            elif time_range == "Last Week":
                filtered_data = stock_data[stock_data.index >= (today - pd.DateOffset(weeks=1))]

            # Plotting the filtered data inside a container with a border and adjusting Y-axis limits
            fig, ax = plt.subplots()
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            ax.plot(filtered_data.index, filtered_data["Close"], label="Close Price")
            ax.set_title(f"{stock_name} ({stock_symbol}) Stock Price", color='white')
            ax.set_xlabel("Date", color='white')
            ax.set_ylabel("Price (USD)", color='white')
            ax.legend()
            ax.grid(True, color='gray')
            ax.set_ylim([filtered_data["Close"].min() * 0.95, filtered_data["Close"].max() * 1.05])  # Adjust Y-axis limits to zoom properly
            ax.tick_params(axis='x', colors='white', rotation=45)
            ax.tick_params(axis='y', colors='white')
            st.pyplot(fig)
    else:
        st.error(f"No stock data available for {stock_name}.")

#endregion

st.write("""
This is not financial advice, just a representation of the time evolution of the price of the stock of some AI-related companies.

The fitting function is chosen based on the assumption, probably wrong, that most AI-related companies are going to thrive in the upcoming years.

As a matter of fact, the graphs are not real-time updated; they update once every day, at market close.
""")
