import streamlit as st

def clear_streamlit_cache():
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        print("Streamlit cache cleared successfully.")
    except Exception as e:
        print(f"Error clearing cache: {e}")

if __name__ == "__main__":
    clear_streamlit_cache()
