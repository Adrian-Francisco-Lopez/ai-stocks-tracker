import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from datetime import datetime, timedelta
import pytz

#### Initializing variables ####
#region

# Access the API key from the secrets.toml file
ALPHA_VANTAGE_API_KEY = st.secrets["alpha_vantage"]["api_key"]
# Constants for stock market hours
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16

# Initialize session state
if "last_data_date" not in st.session_state:
    st.session_state.last_data_date = None
if "stock_data" not in st.session_state:
    st.session_state.stock_data = pd.DataFrame()

#endregion

#### Functions ####
#region

# Get the current Eastern Time (New York Time)
def get_current_eastern_time():
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern)

# Caching stock data with the date of the last update
@st.cache_data(ttl=None)  # No automatic expiry
def get_stock_data_cached(symbol, api_key, current_date):
    """
    This function will fetch and cache stock data if the current date is different from the cached date.
    If it's the same date, it will return cached data.
    """
    # Alpha Vantage API URL
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    
    # Convert data to DataFrame
    if "Time Series (Daily)" in data:
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient="index")
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        })

        # Convert index to datetime and sort by date
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.apply(pd.to_numeric)
        
        return df, current_date  # Cache both the data and the date
    else:
        st.error("Error retrieving data. Please try again later.")
        return pd.DataFrame(), current_date

#endregion

#### Main app logic ####
#region
st.title("AI Companies Stock Tracker")

# Get the current Eastern Time
current_eastern_time = get_current_eastern_time()
current_date = current_eastern_time.date()

# Determine market status
is_market_open = (current_eastern_time.hour > MARKET_OPEN_HOUR or 
                  (current_eastern_time.hour == MARKET_OPEN_HOUR and current_eastern_time.minute >= MARKET_OPEN_MINUTE)) and \
                 (current_eastern_time.hour < MARKET_CLOSE_HOUR)

# Check cache and determine whether to fetch new data
if "last_data_date" not in st.session_state or "stock_data" not in st.session_state:
    # First run or session state empty - fetch data
    st.session_state.stock_data, st.session_state.last_data_date = get_stock_data_cached("NVDA", st.secrets["alpha_vantage"]["api_key"], current_date)
elif st.session_state.last_data_date != current_date and is_market_open:
    # If it's a new date and market is open, refresh data
    st.session_state.stock_data, st.session_state.last_data_date = get_stock_data_cached("NVDA", st.secrets["alpha_vantage"]["api_key"], current_date)
else:
    # Use cached data from previous call
    st.write(f"Using cached data from {st.session_state.last_data_date}")

# Display stock data
nvidia_data = st.session_state.stock_data

# Check if data is available before proceeding
if not nvidia_data.empty:
    # User selection for time range
    time_range = st.selectbox("Select Time Range:", ["Full Range", "Last 5 Years", "Last 3 Years", "Last Year", "Last 6 Months", "Last Month"])

    # Filtering data based on user selection
    today = datetime.now()
    if time_range == "Full Range":
        filtered_data = nvidia_data
    elif time_range == "Last 5 Years":
        filtered_data = nvidia_data[nvidia_data.index >= (today - pd.DateOffset(years=5))]
    elif time_range == "Last 3 Years":
        filtered_data = nvidia_data[nvidia_data.index >= (today - pd.DateOffset(years=3))]
    elif time_range == "Last Year":
        filtered_data = nvidia_data[nvidia_data.index >= (today - pd.DateOffset(years=1))]
    elif time_range == "Last 6 Months":
        filtered_data = nvidia_data[nvidia_data.index >= (today - pd.DateOffset(months=6))]
    elif time_range == "Last Month":
        filtered_data = nvidia_data[nvidia_data.index >= (today - pd.DateOffset(months=1))]

    # Plotting the filtered data
    if "Close" in filtered_data.columns:
        st.line_chart(filtered_data["Close"])
    else:
        st.error("'Close' column not found in the data.")
else:
    st.error("No stock data available to display.")

#endregion

st.write("""This is not financial advice, just a representation of the time evolution of the price of the stock of some AI-related companies.
        //n The fitting function is chosen based on the assumption, probably wrong, that most AI related companies are going to thrive in the upcoming years.
        //n As a matter of fact, the graphs are not real-time updated, they update once every day at market close""")
