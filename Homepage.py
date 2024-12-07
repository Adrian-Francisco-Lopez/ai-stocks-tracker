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
from scipy.stats import norm

#### Initializing variables ####
#region

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

TWELVE_DATA_API_KEY = st.secrets["twelve_data"]["api_key"]

if "stock_data" not in st.session_state:
    st.session_state.stock_data = {}

stocks = {
    "NVIDIA": "NVDA",
    "Advanced Micro Devices": "AMD",
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
    "Applied Materials": "AMAT",
    "Arista": "ANET",
    "Cadence Design Systems": "CDNS",
    "Progress Software": "PRGS",
    "Synopsys": "SNPS",
    "AMAZON": "AMZN"
}

#endregion

#### Fitting ####
#region
def linear_model(x, a, baseline):
    return a * x + baseline

def exponential_model(x, a, b, baseline):
    return a * np.exp(b * x) + baseline

@st.cache_data
def fit_stock_data(x_data, y_data, model_name, p0):
    if model_name == "linear":
        model = linear_model
    elif model_name == "exponential":
        model = exponential_model
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    try:
        params, _ = curve_fit(model, x_data, y_data, p0=p0, maxfev=2000)
        return params
    except RuntimeError as e:
        st.warning(f"An error occurred during fitting ({model_name}): {e}")
        return None

@st.cache_data
def get_fitting_params(symbol, model_type):
    fitting_doc_ref = db.collection("fitting_params").document(f"{symbol}_{model_type}")
    doc = fitting_doc_ref.get()
    if doc.exists:
        params = doc.to_dict().get("params", [])
        start_point = doc.to_dict().get("start_point", 0)  # Default to 0 if not specified
        return params, start_point
    return [], 0

@st.cache_data
def generate_smooth_fit_line(x_data, dates_tuple, params, model_name, num_points=1000):
    if model_name == "linear":
        model = linear_model
    elif model_name == "exponential":
        model = exponential_model
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    x_fine = np.linspace(x_data.min(), x_data.max(), num_points)
    y_fine = model(x_fine, *params)
    dates_numeric = np.array(dates_tuple)
    x_to_date = interp1d(x_data, dates_numeric, kind='linear', bounds_error=False, fill_value='extrapolate')
    dates_fine = pd.to_datetime(x_to_date(x_fine))
    return dates_fine, y_fine

#endregion

#### Functions ####
#region
@st.cache_data(ttl=3600)
def get_stock_data_from_firebase(symbol):
    stock_doc_ref = db.collection("stock_data_json").document(symbol)
    doc = stock_doc_ref.get()
    if doc.exists:
        json_data = doc.to_dict().get("data", "")
        if json_data:
            data = json.loads(json_data)
            df = pd.DataFrame(data)
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df = df.sort_index()
            return df
    return pd.DataFrame()

def classify_stocks_exponential(stock_info_dict):
    buy_stocks = []
    wait_stocks = []
    sell_stocks = []

    for stock_symbol, info in stock_info_dict.items():
        normalized_difference = info["normalized_difference"]
        abs_diff = abs(normalized_difference)
        if abs_diff < 0.04:
            wait_stocks.append((abs_diff, stock_symbol))
        elif normalized_difference < 0:
            buy_stocks.append((abs_diff, stock_symbol))
        else:
            sell_stocks.append((abs_diff, stock_symbol))

    buy_stocks.sort(reverse=True, key=lambda x: x[0])
    wait_stocks.sort(reverse=True, key=lambda x: x[0])
    sell_stocks.sort(reverse=True, key=lambda x: x[0])
    return buy_stocks, wait_stocks, sell_stocks

def classify_stocks_linear(stock_info_dict, time_offset):
    """
    Classify stocks based on linear fit difference for a given time_offset.
    time_offset: a pd.DateOffset defining the timeframe (e.g. pd.DateOffset(years=1))
    """
    buy_stocks = []
    wait_stocks = []
    sell_stocks = []

    today = datetime.now()

    for stock_symbol, info in stock_info_dict.items():
        stock_data = info["stock_data"]
        filtered_data = stock_data[stock_data.index >= (today - time_offset)]

        if not filtered_data.empty and len(filtered_data) > 1:
            # Fit linear model
            x_data = np.arange(len(filtered_data))
            y_data = filtered_data["Close"].values
            slope_guess = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0]) if len(x_data) > 1 else 0.0
            baseline_guess = y_data[0]
            linear_params = fit_stock_data(x_data, y_data, "linear", [slope_guess, baseline_guess])

            if linear_params is not None:
                dates_tuple = tuple(filtered_data.index.view('int64'))
                dates_fine_linear, y_fine_linear = generate_smooth_fit_line(
                    x_data, dates_tuple, linear_params, "linear", num_points=1000
                )
                last_linear_value = y_fine_linear[-1]
                last_close = filtered_data["Close"].iloc[-1]
                linear_diff = (last_close - last_linear_value) / last_close

                abs_diff = abs(linear_diff)
                if abs_diff < 0.04:
                    wait_stocks.append((abs_diff, stock_symbol))
                elif linear_diff < 0:
                    buy_stocks.append((abs_diff, stock_symbol))
                else:
                    sell_stocks.append((abs_diff, stock_symbol))
        else:
            # If we can't fit or no data, put them in wait as a fallback
            wait_stocks.append((0.0, stock_symbol))

    buy_stocks.sort(reverse=True, key=lambda x: x[0])
    wait_stocks.sort(reverse=True, key=lambda x: x[0])
    sell_stocks.sort(reverse=True, key=lambda x: x[0])

    return buy_stocks, wait_stocks, sell_stocks

def plot_full_range_stock(info):
    stock_name = info["stock_name"]
    stock_data = info["stock_data"]
    dates_fine_exp = info["dates_fine_exp"]
    y_fine_exp = info["y_fine_exp"]
    stock_symbol = info["stock_symbol"]

    # Write the stock name
    st.subheader(f"{stock_name} ({stock_symbol}) Stock Price - Full Range")

    fig_full, ax_full = plt.subplots()
    fig_full.patch.set_facecolor('black')
    ax_full.set_facecolor('black')
    ax_full.plot(stock_data.index, stock_data["Close"], label="Close Price")
    # Exponential fit (full range)
    mask_full = (dates_fine_exp >= stock_data.index.min()) & (dates_fine_exp <= stock_data.index.max())
    ax_full.plot(dates_fine_exp[mask_full], y_fine_exp[mask_full], label="Exponential Fit", color='white', linestyle=':')

    ax_full.set_title(f"{stock_symbol} (Full Range)", color='white')
    ax_full.set_xlabel("Date", color='white')
    ax_full.set_ylabel("Price (USD)", color='white')
    ax_full.legend()
    ax_full.grid(True, color='gray')
    ax_full.set_ylim([stock_data["Close"].min() * 0.95, stock_data["Close"].max() * 1.05])
    ax_full.tick_params(axis='x', colors='white', rotation=45)
    ax_full.tick_params(axis='y', colors='white')
    st.pyplot(fig_full)
    plt.close(fig_full)

    # Display last data point info
    st.write(f"**Last data point:** {info['last_date']}  - **Close (Open) Value:** {info['last_value']:.2f}")
    st.write(f"**Last fitted value:** {info['last_fitted_value']:.2f}")
    st.write(f"**Normalized Difference:** {info['normalized_difference']:.2%}")

def plot_short_range_stock(info, time_offset_name, time_offset):
    # Similar to previously done in dropdown, but now directly shown
    stock_name = info["stock_name"]
    stock_data = info["stock_data"]
    dates_fine_exp = info["dates_fine_exp"]
    y_fine_exp = info["y_fine_exp"]
    stock_symbol = info["stock_symbol"]

    # Write the stock name
    st.subheader(f"{stock_name} ({stock_symbol}) - Full Range")

    # Filter data
    today = datetime.now()
    filtered_data = stock_data[stock_data.index >= (today - time_offset)]

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Close Price
    ax.plot(filtered_data.index, filtered_data["Close"], label="Close Price")

    # Exponential Fit in gray (~60%-75% alpha)
    mask = (dates_fine_exp >= filtered_data.index.min()) & (dates_fine_exp <= filtered_data.index.max())
    ax.plot(dates_fine_exp[mask], y_fine_exp[mask], label="Exponential Fit", color='gray', alpha=0.75, linestyle=':')

    # Always show High/Low
    if not filtered_data.empty:
        ax.plot(filtered_data.index, filtered_data["High"], label="High", color='green', linewidth=1)
        ax.plot(filtered_data.index, filtered_data["Low"], label="Low", color='red', linewidth=1)

    linear_params = None
    if len(filtered_data) > 1:
        x_data = np.arange(len(filtered_data))
        y_data = filtered_data["Close"].values
        slope_guess = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0]) if len(x_data) > 1 else 0.0
        baseline_guess = y_data[0]
        linear_params = fit_stock_data(x_data, y_data, "linear", [slope_guess, baseline_guess])

        if linear_params is not None:
            dates_tuple = tuple(filtered_data.index.view('int64'))
            dates_fine_linear, y_fine_linear = generate_smooth_fit_line(
                x_data, dates_tuple, linear_params, "linear", num_points=1000
            )
            ax.plot(dates_fine_linear, y_fine_linear, label="Linear Fit", color='yellow', linestyle=':')

            # Std dev band
            std_dev = filtered_data["Close"].std()
            half_std = std_dev / 2
            ax.fill_between(
                dates_fine_linear,
                y_fine_linear - half_std,
                y_fine_linear + half_std,
                color='gray', alpha=0.45
            )

    ax.set_title(f"{stock_symbol} Stock Price ({time_offset_name})", color='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Price (USD)", color='white')
    ax.grid(True, color='gray')
    if not filtered_data.empty:
        ax.set_ylim([filtered_data["Close"].min() * 0.95, filtered_data["Close"].max() * 1.05])
    ax.tick_params(axis='x', colors='white', rotation=45)
    ax.tick_params(axis='y', colors='white')
    # No legend as per instructions
    st.pyplot(fig)
    plt.close(fig)

    # Display Last Closing Price, Estimated High, Estimated Low, with probabilities
    if not filtered_data.empty and linear_params is not None:
        last_close = filtered_data["Close"].iloc[-1]
        avg_diff_high = (filtered_data["High"] - filtered_data["Close"]).mean()
        avg_diff_low = (filtered_data["Close"] - filtered_data["Low"]).mean()
        estimated_high = last_close + avg_diff_high
        estimated_low = last_close - avg_diff_low

        # Probability calculations
        dates_tuple = tuple(filtered_data.index.view('int64'))
        x_data = np.arange(len(filtered_data))
        y_data = filtered_data["Close"].values
        dates_fine_linear, y_fine_linear = generate_smooth_fit_line(
            x_data, dates_tuple, linear_params, "linear", num_points=1000
        )
        last_linear_value = y_fine_linear[-1]
        std_dev = filtered_data["Close"].std()
        if std_dev > 0:
            Z_high = (estimated_high - last_linear_value) / std_dev
            Z_low = (estimated_low - last_linear_value) / std_dev
            P_high = norm.cdf(Z_high) * 100
            P_low = norm.cdf(Z_low) * 100
        else:
            P_high = 50.0
            P_low = 50.0

        st.write(f"**Normalized Difference:** {(last_close - last_linear_value) / last_close:.2%}")
        st.write(f"**Last closing price:** {last_close:.2f}")
        st.markdown(f"**Estimated Next High Value:** <span style='color:green;'> {estimated_high:.2f}</span> ({P_high:.2f}% probability)", unsafe_allow_html=True)
        st.markdown(f"**Estimated Low Value:** <span style='color:red;'>{estimated_low:.2f}</span> ({P_low:.2f}% probability)", unsafe_allow_html=True)

#endregion

#### Main app logic ####
#region
st.set_page_config(page_title="Stock Tracker")
st.title("Stock Tracker & Estimated Fitting")

# Fetch all stock data and info
stock_info_dict = {}
for stock_name, stock_symbol in stocks.items():
    if stock_symbol not in st.session_state.stock_data:
        st.session_state.stock_data[stock_symbol] = get_stock_data_from_firebase(stock_symbol)

    stock_data = st.session_state.stock_data[stock_symbol]

    if not stock_data.empty:
        stock_data['Close'] = stock_data['Close'].fillna(stock_data['Open'])

        exp_initial_params, start_point_exp = get_fitting_params(stock_symbol, "exponential")
        if exp_initial_params:
            full_x_data = np.arange(len(stock_data))
            full_y_data = stock_data["Close"].values

            x_data_exp = full_x_data[start_point_exp:]
            y_data_exp = full_y_data[start_point_exp:]
            dates_tuple = tuple(stock_data.index[start_point_exp:].view('int64'))
            exp_params = fit_stock_data(x_data_exp, y_data_exp, "exponential", exp_initial_params)
            if exp_params is not None:
                dates_fine_exp, y_fine_exp = generate_smooth_fit_line(
                    x_data_exp, dates_tuple, exp_params, "exponential", num_points=1000
                )

                # Last data point details
                last_data_point = stock_data.iloc[-1]
                last_date = last_data_point.name
                last_value = last_data_point["Close"]
                last_fitted_value = y_fine_exp[-1]
                normalized_difference = (last_value - last_fitted_value) / last_value

                stock_info_dict[stock_symbol] = {
                    "stock_name": stock_name,
                    "stock_data": stock_data,
                    "exp_params": exp_params,
                    "stock_symbol": stock_symbol,
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

# Classification for full range (exponential)
buy_stocks_full, wait_stocks_full, sell_stocks_full = classify_stocks_exponential(stock_info_dict)

# Time offsets for other tabs
time_offsets = {
    "1 Year": pd.DateOffset(years=1),
    "6 Months": pd.DateOffset(months=6),
    "3 Months": pd.DateOffset(months=3),
    "1 Month": pd.DateOffset(months=1),
    "1 Week": pd.DateOffset(weeks=1),
}

# Classification for each timeframe using linear fit
classifications = {}
for name, offset in time_offsets.items():
    classifications[name] = classify_stocks_linear(stock_info_dict, offset)

# Create tabs
tabs = st.tabs(["Full Range", "1 Year", "6 Months", "3 Months", "1 Month", "1 Week"])

# Full range tab: show stocks as per exponential difference classification
with tabs[0]:
    # Show buy_stocks_full
    st.write("#### Below the fit")
    for _, stock_symbol in buy_stocks_full:
        info = stock_info_dict[stock_symbol]
        with st.container(border=True):
            plot_full_range_stock(info)

    st.write("#### Below the fit")
    for _, stock_symbol in wait_stocks_full:
        info = stock_info_dict[stock_symbol]
        with st.container(border=True):
            plot_full_range_stock(info)

    st.write("#### Below the fit")
    for _, stock_symbol in sell_stocks_full:
        info = stock_info_dict[stock_symbol]
        with st.container(border=True):
            plot_full_range_stock(info)

# For other tabs: show stocks classified by linear difference
timeframe_names = ["1 Year", "6 Months", "3 Months", "1 Month", "1 Week"]
for i, tf_name in enumerate(timeframe_names, start=1):
    with tabs[i]:
        buy_stocks_tf, wait_stocks_tf, sell_stocks_tf = classifications[tf_name]

        st.write("#### Below the fit")
        for _, stock_symbol in buy_stocks_tf:
            info = stock_info_dict[stock_symbol]
            with st.container(border=True):
                plot_short_range_stock(info, tf_name, time_offsets[tf_name])

        st.write("#### Equilibrated")
        for _, stock_symbol in wait_stocks_tf:
            info = stock_info_dict[stock_symbol]
            with st.container(border=True):
                plot_short_range_stock(info, tf_name, time_offsets[tf_name])

        st.write("#### Above the fit")
        for _, stock_symbol in sell_stocks_tf:
            info = stock_info_dict[stock_symbol]
            with st.container(border=True):
                plot_short_range_stock(info, tf_name, time_offsets[tf_name])

st.subheader("Disclaimer")
st.write("""
This is not financial advice, just a representation of the time evolution of the price of the stock of some AI-related companies.

The fitting function is chosen based on the assumption, probably wrong, that most AI-related companies are going to thrive in the upcoming years.

As a matter of fact, the graphs are not real-time updated; they update once every day, at market close.
""")

st.subheader("Legend")
st.markdown(
    "<p style='display:inline; color:blue;'>Close Price</p> | "
    "<p style='display:inline; color:green;'>High</p> | "
    "<p style='display:inline; color:red;'>Low</p> | "
    "<p style='display:inline; color:white; opacity:0.6;'>Exponential Fit</p> | "
    "<p style='display:inline; color:yellow;'>Linear Fit</p> | "
    "<p style='display:inline; color:gray;'>Half Standard Deviation</p>",
    unsafe_allow_html=True
)
