import streamlit as st
import requests
import pandas as pd
from datetime import datetime

#### Initializing variables ####
#region

# Access the API key from the secrets.toml file
TWELVE_DATA_API_KEY = st.secrets["twelve_data"]["api_key"]

# Initialize session state
if "stock_data" not in st.session_state:
    st.session_state.stock_data = pd.DataFrame()

#endregion

#### Functions ####
#region

# Fetch stock data from Twelve Data API
def get_stock_data(symbol, api_key):
    """
    This function fetches stock data from the Twelve Data API.
    """
    url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    
    # Convert data to DataFrame
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

        # Convert 'Date' to datetime and sort by date
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by="Date")
        df = df.apply(lambda x: pd.to_numeric(x, errors='ignore') if x.name != 'Date' else x)
        df.set_index("Date", inplace=True)
        
        return df
    else:
        st.error("Error retrieving data. Please try again later.")
        return pd.DataFrame()

#endregion

#### Main app logic ####
#region
# Dictionary of stocks
stocks = {
    "NVIDIA": "NVDA"
}

# Select stock
selected_stock_name = st.selectbox("Select Stock:", list(stocks.keys()))
selected_stock_symbol = stocks[selected_stock_name]

# Title with stock name and symbol
st.title(f"{selected_stock_name} ({selected_stock_symbol}) Stock Tracker")

# Fetch stock data
st.session_state.stock_data = get_stock_data(selected_stock_symbol, TWELVE_DATA_API_KEY)

# Display stock data
stock_data = st.session_state.stock_data

# Check if data is available before proceeding
if not stock_data.empty:
    # User selection for time range
    time_range = st.selectbox("Select Time Range:", ["Full Range", "Last 5 Years", "Last 3 Years", "Last Year", "Last 6 Months", "Last Month"])

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

    # Plotting the filtered data inside a container
    with st.container():
        st.write(f"### {selected_stock_name} ({selected_stock_symbol}) Stock Price")
        st.line_chart(filtered_data["Close"])
else:
    st.error("No stock data available to display.")

#endregion

st.write("""This is not financial advice, just a representation of the time evolution of the price of the stock of some AI-related companies.
        
        The fitting function is chosen based on the assumption, probably wrong, that most AI related companies are going to thrive in the upcoming years.
        
        As a matter of fact, the graphs are not real-time updated, they update once every day, and at market close""")
