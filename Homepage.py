import streamlit as st
import streamlit as st
from alpha_vantage.timeseries import TimeSeries

# Access the API key from the secrets.toml file
ALPHA_VANTAGE_API_KEY = st.secrets["alpha_vantage"]["api_key"]

# Use the key to initialize the TimeSeries object
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

st.title("AI Companies Stock Tracker")
st.write("""This is not financial advice, just a representation of the time evolution of the price of the stock of some AI-related companies.
        /n The fitting function is chosen based on the assumption, probably wrong, that most AI related companies are going to thrive in the upcoming years.
        /n As a matter of fact, the graphs are not real-time updated, they update once every day at market close""")
