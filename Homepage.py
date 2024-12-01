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
    "Broadcom Inc.": "AVGO",
    "NXP Semiconductors": "NXPI",
    "Microsoft": "MSFT",
    "Palantir Technologies": "PLTR",
    "Marvell Technology Inc": "MRVL",
    "Palo Alto Networks": "PANW",
    "Gartner": "IT",
    "Oracle": "ORCL",
    "Service now": "NOW"
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
        st.warning(f"An error occurred during fitting: {e}")
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
@st.cache_data
def generate_smooth_fit_line(x_data, dates_tuple, params, model_name, num_points=1000):
    """
    Generate a smooth line for plotting a fitted model.

    Parameters:
    - x_data: The x-axis data (e.g., days since start).
    - dates_tuple: A tuple of timestamps representing the dates (converted from DatetimeIndex).
    - params: The fitted parameters.
    - model_name: The name of the model ('linear' or 'exponential').
    - num_points: The number of points for the smooth line.

    Returns:
    - dates_fine: The refined dates for the smooth line.
    - y_fine: The fitted values for the smooth line.
    """
    # Map the model name to the corresponding function
    if model_name == "linear":
        model = linear_model
    elif model_name == "exponential":
        model = exponential_model
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # Generate finer x_data
    x_fine = np.linspace(x_data.min(), x_data.max(), num_points)
    # Compute the model values at x_fine
    y_fine = model(x_fine, *params)
    # Map x_fine to dates using interpolation
    dates_numeric = np.array(dates_tuple)  # Convert tuple back to numpy array
    x_to_date = interp1d(x_data, dates_numeric, kind='linear', bounds_error=False, fill_value='extrapolate')
    dates_fine = pd.to_datetime(x_to_date(x_fine))
    return dates_fine, y_fine

#endregion

#### Functions ####
#region

# Fetch stock data from Firebase Firestore
@st.cache_data(ttl=3600) # gets cleared in 1 hours = 3600 seconds
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

def display_stocks_in_tab(stocks_list, tab):
    with tab:
        for _, stock_symbol in stocks_list:  # Unpack abs_diff and stock_symbol
            info = stock_info_dict[stock_symbol]
            stock_name = info["stock_name"]
            stock_data = info["stock_data"]
            dates_fine_exp = info["dates_fine_exp"]
            y_fine_exp = info["y_fine_exp"]

            # Create a container with a border
            with st.container(border=True):
                st.write(f"### {stock_name} ({stock_symbol}) Stock Price - Full Range")

                fig_full, ax_full = plt.subplots()
                fig_full.patch.set_facecolor('black')
                ax_full.set_facecolor('black')
                ax_full.plot(stock_data.index, stock_data["Close"], label="Close Price")

                # Add the exponential fit to the full-range plot
                mask_full = (dates_fine_exp >= stock_data.index.min()) & (dates_fine_exp <= stock_data.index.max())
                ax_full.plot(dates_fine_exp[mask_full], y_fine_exp[mask_full], label="Exponential Fit", color='white', linestyle=':')

                # Set full-range plot details
                ax_full.set_title(f"{stock_name} ({stock_symbol}) Stock Price (Full Range)", color='white')
                ax_full.set_xlabel("Date", color='white')
                ax_full.set_ylabel("Price (USD)", color='white')
                ax_full.legend()
                ax_full.grid(True, color='gray')
                ax_full.set_ylim([stock_data["Close"].min() * 0.95, stock_data["Close"].max() * 1.05])
                ax_full.tick_params(axis='x', colors='white', rotation=45)
                ax_full.tick_params(axis='y', colors='white')
                st.pyplot(fig_full)
                plt.close(fig_full)  # Close the figure to release memory

                # Display last data point information and normalized difference
                st.write(f"""**Last data point:** {info['last_date']}  - **Close value:** {info['last_value']} - **Last fitted value:** {info['last_fitted_value']:.2f}  
                **Normalized Difference:** {info['normalized_difference']:.2%}
                """)

                with st.expander("See Detailed View", expanded=False):

                    # User selection for time range
                    time_range = st.selectbox(
                        f"Select Time Range for {stock_name}:",
                        ["Last 5 Years", "Last 3 Years", "Last Year", "Last 6 Months", "Last 3 Monts", "Last Month", "Last Week"],
                        key=f"{stock_symbol}_time_range",
                        index=3
                    )

                    # Filter data based on user selection
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
                    elif time_range == "Last 3 Months":
                        filtered_data = stock_data[stock_data.index >= (today - pd.DateOffset(months=3))]
                    elif time_range == "Last Month":
                        filtered_data = stock_data[stock_data.index >= (today - pd.DateOffset(months=1))]
                    elif time_range == "Last Week":
                        filtered_data = stock_data[stock_data.index >= (today - pd.DateOffset(weeks=1))]

                    # Plotting
                    fig, ax = plt.subplots()
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')
                    ax.plot(filtered_data.index, filtered_data["Close"], label="Close Price")

                    # Filter the exponential fit to match the filtered data date range and plot
                    mask = (dates_fine_exp >= filtered_data.index.min()) & (dates_fine_exp <= filtered_data.index.max())
                    ax.plot(dates_fine_exp[mask], y_fine_exp[mask], label="Exponential Fit", color='white', linestyle=':')

                    # Set the remaining plot
                    ax.set_title(f"{stock_name} ({stock_symbol}) Stock Price", color='white')
                    ax.set_xlabel("Date", color='white')
                    ax.set_ylabel("Price (USD)", color='white')
                    ax.legend()
                    ax.grid(True, color='gray')
                    ax.set_ylim([filtered_data["Close"].min() * 0.95, filtered_data["Close"].max() * 1.05])
                    ax.tick_params(axis='x', colors='white', rotation=45)
                    ax.tick_params(axis='y', colors='white')
                    st.pyplot(fig)
                    plt.close(fig)  # Close the figure to release memory

#endregion

#### Main app logic ####
#region
# Title of the webpage
st.set_page_config(page_title="Stock Tracker")
st.title("Stock Tracker & Estimated Fitting")

# Initialize lists to categorize stocks
buy_stocks = []
wait_stocks = []
sell_stocks = []

# Dictionary to hold stock information
stock_info_dict = {}

# Main processing loop: Compute everything needed
for stock_name, stock_symbol in stocks.items():
    # Fetch stock data from Firebase
    if stock_symbol not in st.session_state.stock_data:
        st.session_state.stock_data[stock_symbol] = get_stock_data_from_firebase(stock_symbol)

    # Get stock data
    stock_data = st.session_state.stock_data[stock_symbol]

    # Check if data is available before proceeding
    if not stock_data.empty:
        # Use Open price if Close price is not available
        stock_data['Close'] = stock_data['Close'].fillna(stock_data['Open'])

        # Exponential fitting (applies to the entire dataset, from a given start point)
        exp_initial_params, start_point_exp = get_fitting_params(stock_symbol, "exponential")
        if exp_initial_params:
            full_x_data = np.arange(len(stock_data))
            full_y_data = stock_data["Close"].values

            x_data_exp = full_x_data[start_point_exp:]
            y_data_exp = full_y_data[start_point_exp:]
            dates_tuple = tuple(stock_data.index[start_point_exp:].view('int64'))
            exp_params = fit_stock_data(x_data_exp, y_data_exp, "exponential", exp_initial_params)
            if exp_params is not None:
                # Generate exponential fit for plotting
                dates_fine_exp, y_fine_exp = generate_smooth_fit_line(
                    x_data_exp, dates_tuple, exp_params, "exponential", num_points=1000
                )

                # Get the last data point
                last_data_point = stock_data.iloc[-1]
                last_date = last_data_point.name
                last_value = last_data_point["Close"]

                # Get the last fitted value (corresponds to the highest x value in the fit)
                last_fitted_value = y_fine_exp[-1]

                # Compute the normalized difference
                normalized_difference = (last_value - last_fitted_value) / last_value

                # Store all relevant information for later use
                stock_info_dict[stock_symbol] = {
                    "stock_name": stock_name,
                    "stock_data": stock_data,
                    "exp_params": exp_params,
                    "start_point_exp": start_point_exp,
                    "dates_fine_exp": pd.to_datetime(dates_fine_exp),
                    "y_fine_exp": y_fine_exp,
                    "last_date": last_date.strftime("%Y-%m-%d"),
                    "last_value": last_value,
                    "last_fitted_value": last_fitted_value,
                    "normalized_difference": normalized_difference,
                }
            else:
                st.warning(f"Could not fit exponential model for {stock_name}.")
        else:
            st.warning(f"No exponential fitting parameters found for {stock_name}.")
    else:
        st.error(f"No stock data available for {stock_name}.")

# Categorize stocks based on normalized differences
for stock_symbol, info in stock_info_dict.items():
    normalized_difference = info["normalized_difference"]
    abs_diff = abs(normalized_difference)
    if abs_diff < 0.04:
        wait_stocks.append((abs_diff, stock_symbol))
    elif normalized_difference < 0:  # Price below fit => Buy
        buy_stocks.append((abs_diff, stock_symbol))
    else:  # Price above fit => Sell
        sell_stocks.append((abs_diff, stock_symbol))

# Sort stocks in each category by absolute normalized difference in descending order
buy_stocks.sort(reverse=True, key=lambda x: x[0])  # Sort by abs_diff
wait_stocks.sort(reverse=True, key=lambda x: x[0])
sell_stocks.sort(reverse=True, key=lambda x: x[0])

# Create tabs
tab1, tab2, tab3 = st.tabs(["Buy", "Wait", "Sell"])

# Display stocks in each tab
display_stocks_in_tab(buy_stocks, tab1)
display_stocks_in_tab(wait_stocks, tab2)
display_stocks_in_tab(sell_stocks, tab3)
