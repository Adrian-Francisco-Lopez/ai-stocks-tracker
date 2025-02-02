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
    #"Advanced Micro Devices": "AMD", #Quitar, va para abajo
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
    #"Applied Materials": "AMAT", # Quitar, va para abajo
    # Poner estos en algun momento:
    # "GeneDx Holdings Corp": "WGS",
    # "Tempus": "TEM",
    # "ZJK Industrial Co": "ZJK",
    "Arista": "ANET",
    "Cadence Design Systems": "CDNS",
    "Progress Software": "PRGS",
    "Synopsys": "SNPS",
    "AMAZON": "AMZN"
    ""
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
                    x_data, dates_tuple, linear_params, "linear"
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

    st.write(f"### {stock_name} ({stock_symbol}) Stock Price - Full Range")

    fig_full, ax_full = plt.subplots()
    fig_full.patch.set_facecolor('black')
    ax_full.set_facecolor('black')
    ax_full.plot(stock_data.index, stock_data["Close"], label="Close Price")
    # Exponential fit (full range)
    mask_full = (dates_fine_exp >= stock_data.index.min()) & (dates_fine_exp <= stock_data.index.max())
    ax_full.plot(dates_fine_exp[mask_full], y_fine_exp[mask_full], label="Exponential Fit", color='white', linestyle=':')

    ax_full.set_xlabel("Date", color='white')
    ax_full.set_ylabel("Price (USD)", color='white')
    ax_full.legend()
    ax_full.grid(True, color='gray')
    ax_full.set_ylim([stock_data["Close"].min() * 0.95, stock_data["Close"].max() * 1.05])
    ax_full.tick_params(axis='x', colors='white', rotation=45)
    ax_full.tick_params(axis='y', colors='white')
    st.pyplot(fig_full)
    plt.close(fig_full)

    # Display last data point info (rounded)
    st.write(f"**Last data point:** {info['last_date']}  - **Close (Open) Value:** {info['last_value']:.2f}")
    st.write(f"**Last fitted value:** {info['last_fitted_value']:.2f}")
    st.write(f"**Normalized Difference:** {info['normalized_difference']:.2%}")

def plot_short_range_stock(info, time_offset_name, time_offset):
    stock_name = info["stock_name"]
    stock_data = info["stock_data"]
    dates_fine_exp = info.get("dates_fine_exp", np.array([]))
    y_fine_exp = info.get("y_fine_exp", np.array([]))
    stock_symbol = info["stock_symbol"]

    st.write(f"### {stock_name} ({stock_symbol}) Stock Price - {time_offset_name}")

    today = datetime.now()
    filtered_data = stock_data[stock_data.index >= (today - time_offset)]

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Plot Close Price
    ax.plot(filtered_data.index, filtered_data["Close"], label="Close Price")

    # Plot Exponential Fit
    if not filtered_data.empty and len(dates_fine_exp) == len(y_fine_exp):
        # Create mask to filter dates_fine_exp within the filtered_data range
        mask = (dates_fine_exp >= filtered_data.index.min()) & (dates_fine_exp <= filtered_data.index.max())

        # Filter both dates and y values using the mask
        filtered_dates_exp = dates_fine_exp[mask]
        filtered_y_fine_exp = y_fine_exp[mask]

        # Debug: Output lengths
        st.write(f"Exponential Fit - Filtered Dates Length: {len(filtered_dates_exp)}")
        st.write(f"Exponential Fit - Filtered Y Length: {len(filtered_y_fine_exp)}")

        if len(filtered_dates_exp) == len(filtered_y_fine_exp):
            ax.plot(filtered_dates_exp, filtered_y_fine_exp, label="Exponential Fit", color='white', alpha=0.75, linestyle=':')
        else:
            st.warning(f"Length mismatch in exponential fit data for {stock_symbol}. Skipping Exponential Fit.")
    else:
        st.warning(f"Exponential fit dates and values length mismatch for {stock_symbol} in {time_offset_name}.")

    # Plot High/Low
    if not filtered_data.empty:
        ax.plot(filtered_data.index, filtered_data["High"], label="High", color='green', linewidth=1)
        ax.plot(filtered_data.index, filtered_data["Low"], label="Low", color='red', linewidth=1)

    # Plot Linear Fit and Calculate Residuals
    linear_params = None
    residuals = None
    if len(filtered_data) > 1:
        x_data = np.arange(len(filtered_data))
        y_data = filtered_data["Close"].values
        slope_guess = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0]) if len(x_data) > 1 else 0.0
        baseline_guess = y_data[0]
        linear_params = fit_stock_data(x_data, y_data, "linear", [slope_guess, baseline_guess])

        if linear_params is not None:
            # Define two points for the straight line: first and last
            x_start, x_end = x_data[0], x_data[-1]
            y_start, y_end = linear_model(x_start, *linear_params), linear_model(x_end, *linear_params)

            # Plot the straight line
            ax.plot(
                [filtered_data.index[0], filtered_data.index[-1]],
                [y_start, y_end],
                label="Linear Fit",
                color='yellow',
                linestyle=':'
            )
        
        if linear_params is not None:
            dates_tuple = tuple(filtered_data.index.view('int64'))
            dates_fine_linear, y_fine_linear = generate_smooth_fit_line(
                x_data, dates_tuple, linear_params, "linear"
            )
            #ax.plot(dates_fine_linear, y_fine_linear, label="Linear Fit", color='yellow', linestyle=':')

            # Calculate residuals
            y_predicted = linear_model(x_data, *linear_params)
            residuals = y_data - y_predicted

            # Compute standard deviation of residuals
            std_dev_residual = np.std(residuals)

            # Calculate probabilities based on residuals
            if std_dev_residual > 0:
                estimated_high = y_fine_linear[-1] + (filtered_data["High"].iloc[-1] - y_fine_linear[-1])
                estimated_low = y_fine_linear[-1] - (y_fine_linear[-1] - filtered_data["Low"].iloc[-1])

                Z_high = (estimated_high - y_fine_linear[-1]) / std_dev_residual
                Z_low = (estimated_low - y_fine_linear[-1]) / std_dev_residual
                P_high = (1 - norm.cdf(Z_high)) * 100  # Probability above estimated_high
                P_low = norm.cdf(Z_low) * 100          # Probability below estimated_low
            else:
                P_high = 50.0
                P_low = 50.0

            # Fill between using residual-based std_dev with two points
            ax.fill_between(
                [filtered_data.index[0], filtered_data.index[-1]],
                [y_start - std_dev_residual, y_end - std_dev_residual],
                [y_start + std_dev_residual, y_end + std_dev_residual],
                color='gray',
                alpha=0.45
            )
        else:
            st.warning(f"Could not fit linear model for {time_offset_name} in {stock_name}.")

    # Set labels, legend, and style
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Price (USD)", color='white')
    ax.grid(True, color='gray')
    if not filtered_data.empty:
        ax.set_ylim([filtered_data["Close"].min() * 0.95, filtered_data["Close"].max() * 1.05])
    ax.tick_params(axis='x', colors='white', rotation=45)
    ax.tick_params(axis='y', colors='white')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Display last data point info
    if not filtered_data.empty and linear_params is not None:
        last_close = filtered_data["Close"].iloc[-1]
        st.write(f"**Normalized Difference:** {(last_close - y_fine_linear[-1]) / last_close:.2%}")
        st.write(f"**Last closing price:** {last_close:.2f}")
        st.markdown(
            f"**Estimated Next High Value:** <span style='color:green;'> {estimated_high:.2f}</span> ({P_high:.2f}% probability)",
            unsafe_allow_html=True
        )
        st.markdown(
            f"**Estimated Low Value:** <span style='color:red;'>{estimated_low:.2f}</span> ({P_low:.2f}% probability)",
            unsafe_allow_html=True
        )

#### Main app logic ####
#region
st.set_page_config(page_title="Stock Tracker")
st.title("AI-Stock Tracker & Estimated Fitting")

# Fetch all stock data and info
stock_info_dict = {}

# Time ranges for initial predictions
time_ranges = {
    "1 Year": pd.DateOffset(years=1),
    "6 Months": pd.DateOffset(months=6),
    "3 Months": pd.DateOffset(months=3),
    "1 Month": pd.DateOffset(months=1),
    "2 Weeks": pd.DateOffset(weeks=2),
}

for stock_name, stock_symbol in stocks.items():
    if stock_symbol not in st.session_state.stock_data:
        st.session_state.stock_data[stock_symbol] = get_stock_data_from_firebase(stock_symbol)

    stock_data = st.session_state.stock_data[stock_symbol]

    if not stock_data.empty:
        stock_data['Close'] = stock_data['Close'].fillna(stock_data['Open'])

        exp_initial_params, start_point_exp = get_fitting_params(stock_symbol, "exponential")
        if exp_initial_params:

            # Generate exponential fit
            full_x_data = np.arange(len(stock_data))
            full_y_data = stock_data["Close"].values

            x_data_exp = full_x_data[start_point_exp:]
            y_data_exp = full_y_data[start_point_exp:]
            dates_tuple = tuple(stock_data.index[start_point_exp:].view('int64'))
            exp_params = fit_stock_data(x_data_exp, y_data_exp, "exponential", exp_initial_params)

            if exp_params is not None:
                # Initialize session state for predictions
                if stock_symbol not in st.session_state:
                    st.session_state[stock_symbol] = {"predictions": {}}

                dates_fine_exp, y_fine_exp = generate_smooth_fit_line(
                    x_data_exp, dates_tuple, exp_params, "exponential"
                )

                # Prepare last point data
                last_data_point = stock_data.iloc[-1]
                last_date = last_data_point.name
                last_value = last_data_point["Close"]
                last_fitted_value = y_fine_exp[-1]
                normalized_difference = (last_value - last_fitted_value) / last_value

                # Calculate exponential adjustment
                exp_adjustment = last_value - last_fitted_value

                # Prepare exponential fit future
                future_x = np.arange(len(stock_data), len(stock_data) + 365)
                exp_fit_future = exponential_model(future_x, *exp_params) + exp_adjustment

                # Store all information in stock_info_dict
                stock_info_dict[stock_symbol] = {
                    "stock_name": stock_name,
                    "stock_symbol": stock_symbol,
                    "stock_data": stock_data,
                    "exp_params": exp_params,
                    "dates_fine_exp": dates_fine_exp,
                    "y_fine_exp": y_fine_exp,
                    "exp_fit_future": exp_fit_future,
                    "exp_adjustment": exp_adjustment,
                    "last_value": last_value,
                    "last_fitted_value": last_fitted_value,
                    "normalized_difference": normalized_difference,
                    "last_date": last_date.strftime("%Y-%m-%d"),
                }

            else:
                st.warning(f"Could not fit exponential model for {stock_name}.")
        else:
            st.warning(f"No exponential fitting parameters found for {stock_name}.")
    else:
        st.error(f"No stock data available for {stock_name}.")

buy_stocks_full, wait_stocks_full, sell_stocks_full = classify_stocks_exponential(stock_info_dict)

time_offsets = {
    "1 Year": pd.DateOffset(years=1),
    "6 Months": pd.DateOffset(months=6),
    "3 Months": pd.DateOffset(months=3),
    "1 Month": pd.DateOffset(months=1),
    "2 Weeks": pd.DateOffset(weeks=2),
}

classifications = {}
for name, offset in time_offsets.items():
    classifications[name] = classify_stocks_linear(stock_info_dict, offset)

tabs = st.tabs(["Full Range", "1 Year", "6 Months", "3 Months", "1 Month", "2 Weeks"])

with tabs[0]:
    st.subheader("_Below the exponential fit_")
    for _, stock_symbol in buy_stocks_full:
        info = stock_info_dict[stock_symbol]
        with st.container(border=True):
            plot_full_range_stock(info)

    st.subheader("_Equilibrated_")
    for _, stock_symbol in wait_stocks_full:
        info = stock_info_dict[stock_symbol]
        with st.container(border=True):
            plot_full_range_stock(info)

    st.subheader("_Above the exponential fit_")
    for _, stock_symbol in sell_stocks_full:
        info = stock_info_dict[stock_symbol]
        with st.container(border=True):
            plot_full_range_stock(info)

timeframe_names = ["1 Year", "6 Months", "3 Months", "1 Month", "2 Weeks"]
for i, tf_name in enumerate(timeframe_names, start=1):
    with tabs[i]:
        buy_stocks_tf, wait_stocks_tf, sell_stocks_tf = classifications[tf_name]

        st.subheader("_Below the linear fit_")
        for _, stock_symbol in buy_stocks_tf:
            info = stock_info_dict[stock_symbol]
            with st.container(border=True):
                plot_short_range_stock(info, tf_name, time_offsets[tf_name])

        st.subheader("_Equilibrated_")
        for _, stock_symbol in wait_stocks_tf:
            info = stock_info_dict[stock_symbol]
            with st.container(border=True):
                plot_short_range_stock(info, tf_name, time_offsets[tf_name])

        st.subheader("_Above the linear fit_")
        for _, stock_symbol in sell_stocks_tf:
            info = stock_info_dict[stock_symbol]
            with st.container(border=True):
                plot_short_range_stock(info, tf_name, time_offsets[tf_name])

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

st.subheader("Disclaimer")
st.write("""
This is not financial advice, just a representation of the time evolution of the price of the stock of some AI-related companies.

The fitting function is chosen based on the assumption, probably wrong, that most AI-related companies are going to thrive in the upcoming years.

As a matter of fact, the graphs are not real-time updated; they update once every day, at market close.
""")

