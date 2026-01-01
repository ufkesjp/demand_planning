import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime

# --- VERSION CONTROL ---
# Current State: V1.0 - Demand Classification + Model Recommendations
APP_VERSION = "1.0.0" 

def save_versioned_result(df):
    """Saves a background copy of the results for audit/reversion."""
    if not os.path.exists("app_versions"):
        os.makedirs("app_versions")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"app_versions/results_v{APP_VERSION}_{timestamp}.csv"
    df.to_csv(path, index=False)
    return path

# --- Demand Classification Logic ---
def classify_demand(df, item_col, demand_col, date_col, freq_code):
    df[date_col] = pd.to_datetime(df[date_col])
    start_date = df[date_col].min()
    end_date = df[date_col].max()

    def process_item(group):
        resampled = group.set_index(date_col)[demand_col].resample(freq_code).sum()
        full_range = pd.date_range(start=start_date, end=end_date, freq=freq_code)
        series = resampled.reindex(full_range).fillna(0)
        
        non_zero_demand = series[series > 0]
        
        if len(non_zero_demand) == 0:
            return pd.Series({
                'ADI': np.nan, 'CV2': np.nan, 'Category': 'No Demand', 
                'Recommended Forecasting Models': 'N/A'
            })

        adi = len(series) / len(non_zero_demand)
        cv2 = (non_zero_demand.std() / non_zero_demand.mean())**2
        
        if adi < 1.32 and cv2 < 0.49:
            category = 'Smooth'
            models = 'Moving Average, Exponential Smoothing, ARIMA'
        elif adi >= 1.32 and cv2 < 0.49:
            category = 'Intermittent'
            models = "Croston's Method, SBA"
        elif adi < 1.32 and cv2 >= 0.49:
            category = 'Erratic'
            models = 'Holt-Winters, High Safety Stock'
        else:
            category = 'Lumpy'
            models = "Bootstrapping, Manual Override"
            
        return pd.Series({
            'ADI': round(adi, 2), 
            'CV2': round(cv2, 2), 
            'Category': category,
            'Recommended Forecasting Models': models
        })

    results = df.groupby(item_col).apply(process_item).reset_index()
    return results

# --- Streamlit UI ---
st.set_page_config(page_title=f"Demand Planner v{APP_VERSION}", layout="wide")

# Version Display in Top Right
st.markdown(f"<p style='text-align: right; color: gray;'>App Version: {APP_VERSION}</p>", unsafe_allow_html=True)
st.title("📦 Demand Classifier & Forecasting Guide")

with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
    
    # st.divider()
    # st.subheader("Version History")
    # if os.path.exists("app_versions"):
    #     saved_files = os.listdir("app_versions")
    #     for f in sorted(saved_files, reverse=True)[:5]: # Show last 5
    #         st.text(f"📄 {f}")
    # else:
    #     st.caption("No saved versions yet.")

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: item_col = st.selectbox("Item ID Column", options=df.columns)
    with c2: demand_col = st.selectbox("Demand Column", options=df.columns)
    with c3: date_col = st.selectbox("Date Column", options=df.columns)
    with c4: 
        periodicity = st.radio("Periodicity", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True)
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}

    if st.button("Analyze & Version Results"):
        results_df = classify_demand(df, item_col, demand_col, date_col, freq_map[periodicity])
        
        # Save a local background copy for versioning
        saved_path = save_versioned_result(results_df)
        st.toast(f"Result snapshot saved to {saved_path}")

        # Visuals
        st.divider()
        res_c1, res_c2 = st.columns([1, 2])
        with res_c1:
            st.subheader("Summary")
            st.dataframe(results_df['Category'].value_counts().reset_index(), hide_index=True)
        with res_c2:
            fig = px.pie(results_df['Category'].value_counts().reset_index(), values='count', names='Category', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Item-Level Recommendations")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download This Version", data=csv, file_name=f"demand_strategy_v{APP_VERSION}.csv")