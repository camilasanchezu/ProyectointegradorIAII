
"""
PM2.5 Air Quality Forecasting - Streamlit Web Application
Quito, Ecuador
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="PM2.5 Forecasting - Quito",
    page_icon="🌍",
    layout="wide"
)

# Title and description
st.title("🌍 PM2.5 Air Quality Forecasting")
st.markdown("### Quito, Ecuador - Meteorological Stations")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Station selection
stations = ["BELISARIO", "CARAPUNGO", "CENTRO", "COTOCOLLAO", 
            "EL_CAMAL", "SAN_ANTONIO"]
selected_station = st.sidebar.selectbox("📍 Select Station", stations)

# Horizon selection
horizons = {
    "10 Days (Hourly)": "hourly",
    "1 Month (Daily)": "daily", 
    "5 Years (Monthly)": "monthly"
}
selected_horizon = st.sidebar.selectbox("📅 Forecast Horizon", list(horizons.keys()))

# Load data function
@st.cache_data
def load_data():
    """Load and prepare data for the selected station."""
    data_dir = "."

    # Load PM2.5 data
    pm25_df = pd.read_csv(os.path.join(data_dir, "PM2.5.csv"), parse_dates=["Date_time"])

    return pm25_df

# Generate forecast function
def generate_forecast(data, station, horizon_type, forecast_periods):
    """Generate forecast using Prophet model."""

    # Prepare data
    station_col = station.replace("_", " ")
    if station_col not in data.columns:
        station_col = station

    if station_col not in data.columns:
        st.error(f"Station {station} not found in data")
        return None, None

    df_prophet = data[["Date_time", station_col]].copy()
    df_prophet.columns = ["ds", "y"]
    df_prophet = df_prophet.dropna()

    # Resample based on horizon
    if horizon_type == "monthly":
        df_prophet = df_prophet.set_index("ds").resample("M").mean().reset_index()
    elif horizon_type == "daily":
        df_prophet = df_prophet.set_index("ds").resample("D").mean().reset_index()

    df_prophet = df_prophet.dropna()

    if len(df_prophet) < 10:
        st.error("Insufficient data for forecasting")
        return None, None

    # Fit Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=(horizon_type != "monthly"),
        daily_seasonality=(horizon_type == "hourly")
    )
    model.fit(df_prophet)

    # Create future dataframe
    if horizon_type == "monthly":
        freq = "M"
    elif horizon_type == "daily":
        freq = "D"
    else:
        freq = "H"

    future = model.make_future_dataframe(periods=forecast_periods, freq=freq)
    forecast = model.predict(future)

    return model, forecast, df_prophet

# Main content
try:
    data = load_data()

    # Determine forecast periods
    if horizons[selected_horizon] == "hourly":
        periods = 240  # 10 days
    elif horizons[selected_horizon] == "daily":
        periods = 30   # 1 month
    else:
        periods = 60   # 5 years

    # Generate forecast
    with st.spinner("Generating forecast..."):
        result = generate_forecast(data, selected_station, 
                                   horizons[selected_horizon], periods)

    if result[0] is not None:
        model, forecast, original = result

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Components", "📋 Data"])

        with tab1:
            st.subheader(f"PM2.5 Forecast - {selected_station}")

            # Create interactive plot
            fig = go.Figure()

            # Historical data
            fig.add_trace(go.Scatter(
                x=original["ds"], y=original["y"],
                mode="lines", name="Historical",
                line=dict(color="blue", width=1)
            ))

            # Forecast
            future_mask = forecast["ds"] > original["ds"].max()
            fig.add_trace(go.Scatter(
                x=forecast.loc[future_mask, "ds"],
                y=forecast.loc[future_mask, "yhat"],
                mode="lines", name="Forecast",
                line=dict(color="red", width=2)
            ))

            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast.loc[future_mask, "ds"],
                y=forecast.loc[future_mask, "yhat_upper"],
                mode="lines", name="Upper CI",
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast.loc[future_mask, "ds"],
                y=forecast.loc[future_mask, "yhat_lower"],
                mode="lines", name="95% CI",
                fill="tonexty",
                line=dict(width=0),
                fillcolor="rgba(255, 0, 0, 0.2)"
            ))

            # WHO guideline
            fig.add_hline(y=25, line_dash="dash", line_color="orange",
                         annotation_text="WHO Guideline (25 µg/m³)")

            fig.update_layout(
                title=f"PM2.5 Forecast - {selected_station} ({selected_horizon})",
                xaxis_title="Date",
                yaxis_title="PM2.5 (µg/m³)",
                hovermode="x unified",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                avg_historical = original["y"].mean()
                st.metric("Historical Average", f"{avg_historical:.1f} µg/m³")

            with col2:
                avg_forecast = forecast.loc[future_mask, "yhat"].mean()
                delta = avg_forecast - avg_historical
                st.metric("Forecast Average", f"{avg_forecast:.1f} µg/m³",
                         delta=f"{delta:+.1f}")

            with col3:
                max_forecast = forecast.loc[future_mask, "yhat"].max()
                st.metric("Max Forecast", f"{max_forecast:.1f} µg/m³")

            with col4:
                min_forecast = forecast.loc[future_mask, "yhat"].min()
                st.metric("Min Forecast", f"{min_forecast:.1f} µg/m³")

        with tab2:
            st.subheader("Forecast Components")

            # Trend
            fig_trend = px.line(forecast, x="ds", y="trend", 
                               title="Trend Component")
            st.plotly_chart(fig_trend, use_container_width=True)

            # Yearly seasonality
            if "yearly" in forecast.columns:
                fig_yearly = px.line(forecast, x="ds", y="yearly",
                                    title="Yearly Seasonality")
                st.plotly_chart(fig_yearly, use_container_width=True)

        with tab3:
            st.subheader("Forecast Data")

            # Show forecast table
            display_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            display_df.columns = ["Date", "Forecast", "Lower CI", "Upper CI"]
            display_df = display_df[display_df["Date"] > original["ds"].max()]

            st.dataframe(display_df.round(2), use_container_width=True)

            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast Data",
                data=csv,
                file_name=f"pm25_forecast_{selected_station}_{selected_horizon}.csv",
                mime="text/csv"
            )

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please ensure the CSV files are in the same directory as this app.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>PM2.5 Air Quality Forecasting System | Quito, Ecuador</p>
    <p>Data: 2004-2017 | Models: Prophet, LSTM, XGBoost</p>
</div>
""", unsafe_allow_html=True)
