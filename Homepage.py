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
    dates_fine_exp = info["dates_fine_exp"]
    y_fine_exp = info["y_fine_exp"]
    stock_symbol = info["stock_symbol"]

    st.write(f"### {stock_name} ({stock_symbol}) Stock Price - {time_offset_name}")

    today = datetime.now()
    filtered_data = stock_data[stock_data.index >= (today - time_offset)]

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Close Price
    ax.plot(filtered_data.index, filtered_data["Close"], label="Close Price")

    # Exponential Fit in gray (~60%-75% alpha)
    if not filtered_data.empty:
        mask = (dates_fine_exp >= filtered_data.index.min()) & (dates_fine_exp <= filtered_data.index.max())
        ax.plot(dates_fine_exp[mask], y_fine_exp[mask], label="Exponential Fit", color='gray', alpha=0.75, linestyle=':')

        # High/Low
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

            std_dev = filtered_data["Close"].std()
            half_std = std_dev / 2
            ax.fill_between(
                dates_fine_linear,
                y_fine_linear - half_std,
                y_fine_linear + half_std,
                color='gray', alpha=0.45
            )

    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Price (USD)", color='white')
    ax.grid(True, color='gray')
    if not filtered_data.empty:
        ax.set_ylim([filtered_data["Close"].min() * 0.95, filtered_data["Close"].max() * 1.05])
    ax.tick_params(axis='x', colors='white', rotation=45)
    ax.tick_params(axis='y', colors='white')
    st.pyplot(fig)
    plt.close(fig)

    if not filtered_data.empty and linear_params is not None:
        last_close = filtered_data["Close"].iloc[-1]
        avg_diff_high = (filtered_data["High"] - filtered_data["Close"]).mean()
        avg_diff_low = (filtered_data["Close"] - filtered_data["Low"]).mean()
        estimated_high = last_close + avg_diff_high
        estimated_low = last_close - avg_diff_low

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


#### New Functions for Predictions ####
@st.cache_data(ttl=86400)
def smooth_data(y, window_fraction=0.05):
    # A simple way to "smooth" data is a rolling mean.
    # window_fraction of 0.05 means 5% of length as window size
    length = len(y)
    window = max(3, int(length * window_fraction))
    y_smooth = pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().values
    return y_smooth

def sinusoidal_model_fixed_linear(x, A1, f1, phi1, A2, f2, phi2, m, n):
    """
    Sinusoidal model with fixed linear baseline and two sinusoidal components.
    x: Input data
    A1, f1, phi1: Amplitude, frequency, and phase of the first sinusoid
    A2, f2, phi2: Amplitude, frequency, and phase of the second sinusoid
    m, n: Fixed linear parameters
    """
    y = m * x + n  # Linear component
    y += A1 * np.sin(f1 * x + phi1)  # First sinusoid
    y += A2 * np.sin(f2 * x + phi2)  # Second sinusoid
    return y

@st.cache_data(ttl=86400)
def fit_two_sinusoids_fixed_linear(x_data, y_data, std_dev, linear_params):
    """
    Fit the data using a model with fixed linear parameters and two sinusoidal components.
    Parameters:
    - x_data: Independent variable (time or index)
    - y_data: Dependent variable (close price)
    - std_dev: Standard deviation of the data (used for initial amplitude guesses)
    - linear_params: Tuple of fixed linear parameters (m, n)

    Returns:
    - Fitted parameters for the sinusoidal components, or None if the fit fails.
    """
    m, n = linear_params  # Fixed linear parameters

    # Initial guesses for sinusoidal parameters
    A1_guess = 2 * std_dev  # Amplitude of the first sinusoid
    f1_guess = 2 * np.pi / len(x_data)  # Frequency of the first sinusoid
    phi1_guess = np.pi / 2  # Phase of the first sinusoid

    A2_guess = std_dev  # Amplitude of the second sinusoid
    f2_guess = 4 * np.pi / len(x_data)  # Frequency of the second sinusoid
    phi2_guess = 0  # Phase of the second sinusoid

    p0 = [A1_guess, f1_guess, phi1_guess, A2_guess, f2_guess, phi2_guess]  # Initial guesses

    try:
        # Fit the model, keeping m and n fixed
        popt, _ = curve_fit(
            lambda x, A1, f1, phi1, A2, f2, phi2: sinusoidal_model_fixed_linear(x, A1, f1, phi1, A2, f2, phi2, m, n),
            x_data,
            y_data,
            p0=p0,
            bounds=(
                [0, 0, -np.pi, 0, 0, -np.pi],  # Lower bounds
                [np.inf, np.inf, np.pi, np.inf, np.inf, np.pi],  # Upper bounds
            ),
            maxfev=10000
        )
        return popt  # Return optimized sinusoidal parameters
    except RuntimeError:
        return None  # Return None if the fitting fails

def plot_prediction_chart(stock_data, stock_symbol, stock_name):
    time_ranges = {
        "1 Year": pd.DateOffset(years=1),
        "6 Months": pd.DateOffset(months=6),
        "3 Months": pd.DateOffset(months=3),
        "1 Month": pd.DateOffset(months=1),
        "2 Weeks": pd.DateOffset(weeks=2),
    }

    with st.expander("Stock evolution prediction"):
        for label, offset in time_ranges.items():
            today = datetime.now()
            filtered_data = stock_data[stock_data.index >= (today - offset)]
            if filtered_data.empty:
                st.write(f"No data for {label}")
                continue

            x_data = np.arange(len(filtered_data))
            y_data = filtered_data["Close"].values
            std_dev = filtered_data["Close"].std()

            fig, ax = plt.subplots()
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

            # Original close price
            ax.plot(filtered_data.index, y_data, label="Close Price")

            # Smoothed close price
            y_smooth = smooth_data(y_data, window_fraction=0.1)
            ax.plot(filtered_data.index, y_smooth, label="Smoothed Close", color='blue', alpha=0.5)

            # Compute linear fit parameters for this time range
            if len(y_data) > 1:
                slope_guess = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0])
            else:
                slope_guess = 0.0
            baseline_guess = y_data[0]
            linear_params = fit_stock_data(x_data, y_data, "linear", [slope_guess, baseline_guess])
            if linear_params is None:
                st.write("Could not fit linear model for this timeframe")
                plt.close(fig)
                continue

            # Fit two sinusoids with fixed linear parameters
            sinusoidal_params = fit_two_sinusoids_fixed_linear(x_data, y_data, std_dev, linear_params)
            if sinusoidal_params is not None:
                # Extend prediction
                future_len = int(len(x_data) * 1.5)
                x_future = np.arange(future_len)
                y_fit = sinusoidal_model_fixed_linear(x_future, *sinusoidal_params, *linear_params)

                ax.plot(filtered_data.index, y_fit[:len(x_data)], label="Sinusoidal Fit", color='yellow')
                future_dates = pd.date_range(filtered_data.index[-1], periods=future_len-len(x_data)+1, freq='D')[1:]
                ax.plot(future_dates, y_fit[len(x_data):], linestyle='--', color='yellow', alpha=0.7, label="Prediction")
            else:
                st.write("Fit did not converge, could not predict")

            ax.set_title(f"{stock_symbol} Stock {label} Prediction", color='white')
            ax.set_xlabel("Date", color='white')
            ax.set_ylabel("Price (USD)", color='white')
            ax.grid(True, color='gray')
            ax.tick_params(axis='x', colors='white', rotation=45)
            ax.tick_params(axis='y', colors='white')

            st.pyplot(fig)
            plt.close(fig)

@st.cache_data(ttl=86400)
def calculate_weighted_prediction(predictions, weights):
    """
    Calculate the weighted average of predicted data.
    Args:
        predictions (dict): Keys are time ranges (e.g., "1 Year") and values are predicted arrays.
        weights (dict): Keys are time ranges and values are weights.

    Returns:
        np.array: Weighted averaged predictions for the future.
    """
    total_weight = sum(weights.values())
    if total_weight == 0:
        st.warning("Total weight is zero. Returning NaN array.")
        return np.full_like(next(iter(predictions.values())), np.nan)
    
    # Initialize with zeros of the same shape as the first prediction
    weighted_sum = np.zeros_like(predictions[next(iter(predictions))])  # Initialize to the shape of a prediction

    for time_range, prediction in predictions.items():
        weight = weights[time_range]
        weighted_sum += weight * prediction

    # Normalize to the total weight
    weighted_average = weighted_sum / total_weight
    return weighted_average

def prepare_overall_predictions_for_stock(info, weights_dict):
    """
    Prepare weighted predictions for the Overall tab using specified weight dictionaries.

    Args:
        info (dict): Stock info containing predicted sinusoidal and exponential fits.
        weights_dict (dict): Dictionary of weights for each time range, including 'Exponential'.

    Returns:
        dict: Weighted averaged predictions for each final horizon.
    """
    predictions = {}

    # Extract sinusoidal predictions for all time ranges
    for time_range in ["1 Year", "6 Months", "3 Months", "1 Month", "2 Weeks"]:
        key = f"predicted_{time_range.replace(' ', '_').lower()}"
        if key in info["predicted_fits"]:
            predictions[time_range] = info["predicted_fits"][key]
        else:
            st.warning(f"Missing {key} for {info['stock_symbol']}. Filling with NaN.")
            predictions[time_range] = np.full(future_horizons[time_range], np.nan)

    # Add Exponential predictions
    if "exp_fit_future" in info and "exp_adjustment" in info:
        predictions["Exponential"] = info["exp_fit_future"][:max(future_horizons.values())] + info["exp_adjustment"]
    else:
        st.warning(f"Missing Exponential predictions for {info['stock_symbol']}. Filling with NaN.")
        predictions["Exponential"] = np.full(max(future_horizons.values()), np.nan)

    # Calculate weighted predictions for each final horizon
    final_predictions = {}

    for horizon, weights in weights_dict.items():
        # Calculate weighted average
        weighted_pred = calculate_weighted_prediction(predictions, weights)
        final_predictions[horizon] = weighted_pred[:future_horizons[horizon]]

    return final_predictions

#endregion

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

# Future horizons for final predictions
future_horizons = {
    "1 Year": 365,     # Extend by 365 days
    "6 Months": 182,   # Extend by approximately 6 months
    "3 Months": 91,    # Extend by approximately 3 months
    "1 Month": 30,     # Extend by 1 month
    "2 Weeks": 14,     # Extend by 2 weeks
}

# Define separate weight dictionaries for the Overall tab, including 'Exponential'
horizon_weights_overall_2_weeks = {
    "1 Year": 1,
    "6 Months": 2,
    "3 Months": 4,
    "1 Month": 8,
    "2 Weeks": 16,
    "Exponential": 2  # Added weight for Exponential
}

horizon_weights_overall_3_months = {
    "1 Year": 2,
    "6 Months": 4,
    "3 Months": 8,
    "1 Month": 16,
    "2 Weeks": 8,
    "Exponential": 2  # Added weight for Exponential
}

horizon_weights_overall_1_year = {
    "1 Year": 16,
    "6 Months": 8,
    "3 Months": 4,
    "1 Month": 2,
    "2 Weeks": 1,
    "Exponential": 2  # Added weight for Exponential
}

# Define the weights dictionary for the Overall tab
weights_dict = {
    "2 Weeks": horizon_weights_overall_2_weeks,
    "3 Months": horizon_weights_overall_3_months,
    "1 Year": horizon_weights_overall_1_year,
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
                dates_fine_exp, y_fine_exp = generate_smooth_fit_line(
                    x_data_exp, dates_tuple, exp_params, "exponential", num_points=1000
                )

                # Prepare last point data
                last_data_point = stock_data.iloc[-1]
                last_date = last_data_point.name
                last_value = last_data_point["Close"]
                last_fitted_value = y_fine_exp[-1]
                normalized_difference = (last_value - last_fitted_value) / last_value


                # Prepare exponential fit future
                future_x = np.arange(len(stock_data), len(stock_data) + max(future_horizons.values()))
                exp_fit_future = exponential_model(future_x, *exp_params)
                #exp_adjustment = last_value - exponential_model(len(x_data_exp) - 1, *exp_params)

                # Calculate exponential adjustment
                exp_adjustment = last_value - last_fitted_value

                # Initialize storage for time range predictions
                predicted_fits = {}
                for time_range, offset in time_ranges.items():
                    # Filter data for each time range
                    filtered_data = stock_data[stock_data.index >= (datetime.now() - offset)]
                    if len(filtered_data) > 1:
                        x_data = np.arange(len(filtered_data))
                        y_data = filtered_data["Close"].values
                        std_dev = filtered_data["Close"].std()

                        # Fit linear model for this time range
                        slope_guess = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0]) if len(x_data) > 1 else 0.0
                        baseline_guess = y_data[0]
                        linear_params = fit_stock_data(x_data, y_data, "linear", [slope_guess, baseline_guess])

                        if linear_params is not None and len(linear_params) > 0:
                            # Fit two sinusoids
                            sinusoidal_params = fit_two_sinusoids_fixed_linear(x_data, y_data, std_dev, linear_params)
                            if sinusoidal_params is not None:
                                # Generate predictions for the future horizon
                                future_len = future_horizons[time_range]
                                x_future_pred = np.arange(len(x_data), len(x_data) + future_len)
                                sinusoidal_prediction = sinusoidal_model_fixed_linear(
                                    x_future_pred, *sinusoidal_params, *linear_params
                                )
                                predicted_fits[f"predicted_{time_range.replace(' ', '_').lower()}"] = sinusoidal_prediction
                            else:
                                st.warning(f"Could not fit sinusoids for {time_range} in {stock_name}. Filling with NaN.")
                                predicted_fits[f"predicted_{time_range.replace(' ', '_').lower()}"] = np.full(future_horizons[time_range], np.nan)
                        else:
                            st.warning(f"Could not fit linear model for {time_range} in {stock_name}. Filling with NaN.")
                            predicted_fits[f"predicted_{time_range.replace(' ', '_').lower()}"] = np.full(future_horizons[time_range], np.nan)
                    else:
                        st.warning(f"Not enough data for {time_range} in {stock_name}. Filling with NaN.")
                        predicted_fits[f"predicted_{time_range.replace(' ', '_').lower()}"] = np.full(future_horizons[time_range], np.nan)

                            #if sinusoidal_params is not None and len(sinusoidal_params) > 0:
                                # Generate predictions for the future horizon
                                #if time_range not in future_horizons:
                                    #st.error(f"Time range '{time_range}' is missing from future_horizons.")
                                    #continue
                                #future_x = np.arange(len(x_data), len(x_data) + future_horizons[time_range])
                                #predicted_fit = sinusoidal_model_fixed_linear(future_x, *sinusoidal_params, *linear_params)
                                #predicted_fits[f"predicted_{time_range.replace(' ', '_').lower()}"] = sinusoidal_prediction_array
                                #predicted_fits[f"predicted_{time_range.replace(' ', '_').lower()}"] = (
                                    #predicted_fit if predicted_fit is not None else np.full(future_horizons[time_range], np.nan)
                                #)

                ######## Debugging and verification ###########
                if not predicted_fits:
                    st.warning(f"No predictions generated for {stock_name} ({stock_symbol}).")

                # Verify all expected keys
                expected_time_ranges = [f"predicted_{time_range.replace(' ', '_').lower()}" for time_range in time_ranges]
                for key in expected_time_ranges:
                    if key not in predicted_fits:
                        st.warning(f"Prediction for {key} in {stock_symbol} is missing. Filling with NaN.")
                        time_range_key = key.replace("predicted_", "").replace("_", " ").title()
                        predicted_fits[key] = np.full(future_horizons[time_range_key], np.nan)

                # Verify exponential fit
                if "exp_fit_future" not in locals() or exp_fit_future is None:
                    st.warning(f"Exponential fit future missing for {stock_name} ({stock_symbol}). Filling with NaN.")
                    exp_fit_future = np.full(max(future_horizons.values()), np.nan)

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
                    "predicted_fits": predicted_fits,
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

tabs = st.tabs(["Full Range", "1 Year", "6 Months", "3 Months", "1 Month", "2 Weeks", "Predictions", "Overall"])

with tabs[0]:
    st.subheader("_Below the fit_")
    for _, stock_symbol in buy_stocks_full:
        info = stock_info_dict[stock_symbol]
        with st.container(border=True):
            plot_full_range_stock(info)

    st.subheader("_Equilibrated_")
    for _, stock_symbol in wait_stocks_full:
        info = stock_info_dict[stock_symbol]
        with st.container(border=True):
            plot_full_range_stock(info)

    st.subheader("_Above the fit_")
    for _, stock_symbol in sell_stocks_full:
        info = stock_info_dict[stock_symbol]
        with st.container(border=True):
            plot_full_range_stock(info)

timeframe_names = ["1 Year", "6 Months", "3 Months", "1 Month", "2 Weeks"]
for i, tf_name in enumerate(timeframe_names, start=1):
    with tabs[i]:
        buy_stocks_tf, wait_stocks_tf, sell_stocks_tf = classifications[tf_name]

        st.subheader("_Below the fit_")
        for _, stock_symbol in buy_stocks_tf:
            info = stock_info_dict[stock_symbol]
            with st.container(border=True):
                plot_short_range_stock(info, tf_name, time_offsets[tf_name])

        st.subheader("_Equilibrated_")
        for _, stock_symbol in wait_stocks_tf:
            info = stock_info_dict[stock_symbol]
            with st.container(border=True):
                plot_short_range_stock(info, tf_name, time_offsets[tf_name])

        st.subheader("_Above the fit_")
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

with tabs[6]:  # Predictions tab
    for stock_symbol, info in stock_info_dict.items():
        # Compute or retrieve the linear fit parameters
        stock_data = info["stock_data"]
        x_data = np.arange(len(stock_data))
        y_data = stock_data["Close"].values

        # Compute linear fit parameters if not available
        slope_guess = (y_data[-1] - y_data[0]) / (x_data[-1] - x_data[0]) if len(x_data) > 1 else 0.0
        baseline_guess = y_data[0]
        linear_params = fit_stock_data(x_data, y_data, "linear", [slope_guess, baseline_guess])

        if linear_params is None:
            st.warning(f"Could not compute linear fit for {info['stock_name']} ({stock_symbol}). Skipping prediction.")
            continue

        # Pass linear_params to the prediction chart
        with st.container(border=True):
            st.subheader(f"{info['stock_name']} ({stock_symbol})")
            plot_prediction_chart(info["stock_data"], stock_symbol, info['stock_name'])

with tabs[7]:  # Overall tab
    st.write("### Overall Predictions")

    # Define the final horizons for the Overall tab
    overall_final_horizons = ["2 Weeks", "3 Months", "1 Year"]

    for stock_symbol, info in stock_info_dict.items():
        # Prepare final predictions using the Overall weight dictionaries
        final_preds = prepare_overall_predictions_for_stock(info, {
            "2 Weeks": horizon_weights_overall_2_weeks,
            "3 Months": horizon_weights_overall_3_months,
            "1 Year": horizon_weights_overall_1_year,
        })

        with st.container(border=True):
            st.subheader(f"{info['stock_name']} ({stock_symbol})")
            with st.expander("Weighted Average Predictions"):
                for horizon, color in zip(overall_final_horizons, ["yellow", "green", "blue"]):
                    if horizon in final_preds:
                        prediction = final_preds[horizon]
                        if not np.all(np.isnan(prediction)):
                            fig, ax = plt.subplots()
                            fig.patch.set_facecolor('black')
                            ax.set_facecolor('black')
                            ax.plot(prediction, label=f"{horizon} Prediction", color=color)
                            ax.set_title(f"{stock_symbol} - {horizon} Prediction", color='white')
                            ax.set_xlabel("Days", color='white')
                            ax.set_ylabel("Price (USD)", color='white')
                            ax.tick_params(axis='x', colors='white')
                            ax.tick_params(axis='y', colors='white')
                            ax.grid(color='gray', alpha=0.5)
                            ax.legend()
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.warning(f"{horizon} Prediction for {stock_symbol} contains only NaN values.")
                    else:
                        st.warning(f"{horizon} Prediction for {stock_symbol} is missing.")

st.subheader("Disclaimer")
st.write("""
This is not financial advice, just a representation of the time evolution of the price of the stock of some AI-related companies.

The fitting function is chosen based on the assumption, probably wrong, that most AI-related companies are going to thrive in the upcoming years.

As a matter of fact, the graphs are not real-time updated; they update once every day, at market close.
""")

