import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from src.data_processor import process_telemetry_batch, calculate_vhi_and_status

# Streamlit Page Setup
st.set_page_config(
    page_title="Vehicle Health Prediction Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Health Prediction & Telematics Dashboard")
st.markdown("Monitor real-time telematics parameters, battery health index (VHI), and anomaly states.")

# Sidebar Controls
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Telemetry JSON", type=["json"])

@st.cache_data
def load_default_data():
    # Fallback to demo data if no file uploaded
    sample_record = [{
        'vehicleId': 'MAT561004P0A00021',
        'eventDateTime': '2026-03-21T00:00:58.000000972',
        'speed': 0, 'odometer': 208520, 'accelX': -0.87, 'accelY': -0.87, 'accelZ': 20.45,
        'vehicleBatteryVoltage': 24.98, 'signalStrength': 31
    }]
    return pd.DataFrame(sample_record)

if uploaded_file is not None:
    raw_json = [json.loads(line) for line in uploaded_file.getvalue().decode("utf-8").splitlines() if line.strip()]
    df = process_telemetry_batch(raw_json)
    df = calculate_vhi_and_status(df)
else:
    st.info("💡 Upload a telematics JSON file or use generated preview data.")
    df = load_default_data()
    df = calculate_vhi_and_status(df)

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
if 'vhi' in df.columns:
    latest_vhi = df['vhi'].iloc[0] if not df.empty else 0
    latest_status = df['health_status'].iloc[0] if not df.empty else 'N/A'
    latest_voltage = df['vehicleBatteryVoltage'].iloc[0] if not df.empty else 0
    total_records = len(df)

    col1.metric("Current VHI Score", f"{latest_vhi:.1f} / 100")
    col2.metric("Health Status", latest_status)
    col3.metric("Battery Voltage", f"{latest_voltage:.2f} V")
    col4.metric("Telemetry Batches Processed", f"{total_records:,}")

st.divider()

# Interactive Visualizations
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("⚡ Battery Voltage & VHI Over Time")
    if 'eventDateTime' in df.columns and 'vehicleBatteryVoltage' in df.columns:
        fig_vhi = px.line(
            df, 
            x='eventDateTime', 
            y=['vehicleBatteryVoltage', 'vhi'],
            title="Battery & VHI Dynamics",
            labels={'value': 'Value', 'eventDateTime': 'Timestamp'}
        )
        st.plotly_chart(fig_vhi, use_container_width=True)

with right_col:
    st.subheader("📊 Health Status Distribution")
    if 'health_status' in df.columns:
        status_counts = df['health_status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig_pie = px.pie(
            status_counts, 
            names='Status', 
            values='Count', 
            color='Status',
            color_discrete_map={'Healthy': '#2ecc71', 'Degrading': '#f39c12', 'Critical': '#e74c3c'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# Data Explorer Section
with st.expander("🔍 View Processed Telematics Data"):
    st.dataframe(df, use_container_width=True)
