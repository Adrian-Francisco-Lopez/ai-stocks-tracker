import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials, firestore
import json
from scipy.optimize import curve_fit
import numpy as np
from scipy.interpolate import interp1d

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
    "Meta Platforms": "META",
    "Palantir Technologies": "PLTR",
    "Marvell Technology Inc": "MRVL",
    "CrowdStrike": "CRWD"
}

#endregion

#### Fitting ####
#region
# Define different fitting functions
def linear_model(x, a, baseline):
    return a * x + baseline

def exponential_model(x, a, b, baseline):
    return a * np.exp(b * x) + baseline

def logistic_model(x, L, k, x0, baseline):
    return L / (1 + np.exp(-k * (x - x0))) + baseline

@st.cache_data
def fit_stock_data(x_data, y_data, model_name, p0):
    """
    Fit stock data using a specified model.

    Parameters:
    - x_data: The x-axis data (e.g., days since start).
    - y_data: The y-axis data (e.g., stock close values).
    - model_name: The name of the fitting model ('linear' or 'exponential').
    - p0: Initial parameter estimates (list).

    Returns:
    - Fitted parameters (list).
    """
    # Map the model name to the corresponding function
    if model_name == "linear":
        model = linear_model
    elif model_name == "exponential":
        model = exponential_model
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # Perform the fitting
    try:
        params, _ = curve_fit(model, x_data, y_data, p0=p0, maxfev=2000)
        return params
    except RuntimeError as e:
        print(f"An error occurred during fitting: {e}")
        return None

# Fetch fitting parameters from Firebase Firestore
@st.cache_data
def get_fitting_params(symbol, model_type):
    """
    Fetch fitting parameters from Firebase Firestore for a given stock and model type.
    """
    fitting_doc_ref = db.collection("fitting_params").document(f"{symbol}_{model_type}")
    doc = fitting_doc_ref.get()
    if doc.exists:
        params = doc.to_dict().get("params", [])
        start_point = doc.to_dict().get("start_point", 0)  # Default to 0 if not specified
        return params, start_point
    return [], 0

# Smooth bumpy lines by interpolating data
def generate_smooth_fit_line(x_data, dates, params, model, num_points=1000):
    # Generate finer x_data
    x_fine = np.linspace(x_data.min(), x_data.max(), num_points)
    # Compute the model values at x_fine
    y_fine = model(x_fine, *params)
    # Map x_fine to dates using interpolation
    #dates_numeric = dates.astype(np.int64)
    dates_numeric = dates.view('int64')  # Safe conversion to int64
    x_to_date = interp1d(x_data, dates_numeric, kind='linear', bounds_error=False, fill_value='extrapolate')
    dates_fine = pd.to_datetime(x_to_date(x_fine))
    return dates_fine, y_fine

#endregion


#### Functions ####
#region

# Fetch stock data from Firebase Firestore
@st.cache_data
def get_stock_data_from_firebase(symbol):
    """
    This function fetches stock data from Firebase Firestore as a JSON document.
    """
    # Firestore reference for the JSON document of each stock
    stock_doc_ref = db.collection("stock_data_json").document(symbol)

    # Fetch the document
    doc = stock_doc_ref.get()
    if doc.exists:
        json_data = doc.to_dict().get("data", "")
        if json_data:
            # Load JSON data into DataFrame
            data = json.loads(json_data)
            df = pd.DataFrame(data)
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df = df.sort_index()
            return df
    return pd.DataFrame()

#endregion

#### Main app logic ####
#region

# Title of the webpage
st.set_page_config(page_title="Stock Tracker")
st.title("Stock Tracker & estimated fitting")

# Loop through all stocks
for stock_name, stock_symbol in stocks.items():
    # Fetch stock data from Firebase
    if stock_symbol not in st.session_state.stock_data:
        st.session_state.stock_data[stock_symbol] = get_stock_data_from_firebase(stock_symbol)

    # Display stock data
    stock_data = st.session_state.stock_data[stock_symbol]

    # Check if data is available before proceeding
    if not stock_data.empty:
        # Use Open price if Close price is not available
        stock_data['Close'] = stock_data['Close'].fillna(stock_data['Open'])

        with st.container(border=True):
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

            # Prepare data for fitting
            x_data = np.arange(len(filtered_data))
            y_data = filtered_data["Close"].values

            # Linear fitting (applies to filtered data based on selected time range, excluding Full Range, Last 5 Years, Last 3 Years)
            if time_range in ["Last Year", "Last 6 Months", "Last Month", "Last Week"]:
                linear_initial_params, start_point_linear = get_fitting_params(stock_symbol, "linear")
                if linear_initial_params:
                    # Fit using the fit_stock_data function on the filtered data
                    linear_params = fit_stock_data(x_data, y_data, "linear", linear_initial_params)

                    # Generate smooth fit line for plotting
                    dates_fine_linear, y_fine_linear = generate_smooth_fit_line(x_data, filtered_data.index, linear_params, linear_model, 4)

            # Exponential fitting (applies to the entire dataset, from a given start point)
            # Here we fetch the full dataset, rather than filtered_data, to make sure it spans all available data
            full_x_data = np.arange(len(stock_data))
            full_y_data = stock_data["Close"].values

            exp_initial_params, start_point_exp = get_fitting_params(stock_symbol, "exponential")
            if exp_initial_params:
                
                # Slice the full data starting from start_point_exp
                x_data_exp = full_x_data[start_point_exp:]
                y_data_exp = full_y_data[start_point_exp:]
                
                # Fit using the fit_stock_data function
                exp_params = fit_stock_data(x_data_exp, y_data_exp, "exponential", exp_initial_params)

                # Generate smooth fit line for plotting
                dates_fine_exp, y_fine_exp = generate_smooth_fit_line(x_data_exp, stock_data.index[start_point_exp:], exp_params, exponential_model, 1000)

            # Plotting the filtered data inside a container with a border and adjusting Y-axis limits
            fig, ax = plt.subplots()
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            ax.plot(filtered_data.index, filtered_data["Close"], label="Close Price")
            
            if 'dates_fine_linear' in locals() and time_range in ["Last Year", "Last 6 Months", "Last Month", "Last Week"]:
                ax.plot(dates_fine_linear, y_fine_linear, label="Linear Fit", color='yellow', linestyle=':')
            if 'dates_fine_exp' in locals():
                # Filter the exponential fit to the date range of filtered_data
                mask = (dates_fine_exp >= filtered_data.index.min()) & (dates_fine_exp <= filtered_data.index.max())
                ax.plot(dates_fine_exp[mask], y_fine_exp[mask], label="Exponential Fit", color='white', linestyle=':')

            # set the remaining plot
            ax.set_title(f"{stock_name} ({stock_symbol}) Stock Price", color='white')
            ax.set_xlabel("Date", color='white')
            ax.set_ylabel("Price (USD)", color='white')
            ax.legend()
            ax.grid(True, color='gray')
            ax.set_ylim([filtered_data["Close"].min() * 0.95, filtered_data["Close"].max() * 1.05])  # Adjust Y-axis limits to zoom properly
            ax.tick_params(axis='x', colors='white', rotation=45)
            ax.tick_params(axis='y', colors='white')
            st.pyplot(fig)

            # Display last data point information
            last_data_point = filtered_data.iloc[-1]
            last_date = last_data_point.name.strftime("%Y-%m-%d")
            last_value = last_data_point["Close"]
            value_type = "Close value" if "Close" in filtered_data.columns and not pd.isna(last_data_point["Close"]) else "Open value"
            st.write(f"**Last data point:** {last_date} - {value_type}: {last_value}")
    else:
        st.error(f"No stock data available for {stock_name}.")

#endregion

st.write("""
This is not financial advice, just a representation of the time evolution of the price of the stock of some AI-related companies.

The fitting function is chosen based on the assumption, probably wrong, that most AI-related companies are going to thrive in the upcoming years.

As a matter of fact, the graphs are not real-time updated; they update once every day, at market close.
""")
