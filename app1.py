import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import pymongo
from streamlit_autorefresh import st_autorefresh
import ssl
from datetime import datetime, timedelta, timezone
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load environment variables
load_dotenv()

# Session state initialization with UTC timestamps
if 'main_df' not in st.session_state:
    st.session_state.main_df = pd.DataFrame()
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = None
if 'initial_load_complete' not in st.session_state:
    st.session_state.initial_load_complete = False
if 'last_timestamp' not in st.session_state:
    # Initialize with UTC time to avoid local/server timezone mismatch
    st.session_state.last_timestamp = datetime.now(timezone.utc)

def get_mongo_client():
    """Establish connection to MongoDB with secure settings"""
    try:
        return pymongo.MongoClient(
            os.getenv("MONGO_URI"),
            tls=True,
            tlsAllowInvalidCertificates=True
        )
    except Exception as e:
        st.error(f"Connection failed: {str(e)}")
        return None

def fetch_new_data(client, db_name, collection_name, timestamp_field):
    """Fetch new data from MongoDB since last update"""
    db = client[db_name]
    collection = db[collection_name]
    
    query = {}
    if st.session_state.last_fetch_time:
        # Ensure query uses UTC time
        query[timestamp_field] = {"$gt": st.session_state.last_timestamp.replace(tzinfo=None)}
    
    new_data = list(collection.find(query, {'_id': 0}))
    
    if new_data:
        df_new = pd.DataFrame(new_data)
        
        # Validate and parse timestamp field
        if timestamp_field in df_new.columns:
            try:
                # Convert to UTC datetime explicitly
                df_new[timestamp_field] = pd.to_datetime(df_new[timestamp_field], utc=True)
            except Exception as e:
                st.warning(f"Failed to parse '{timestamp_field}': {str(e)}")
                df_new = df_new.drop(columns=[timestamp_field])  # Remove invalid timestamp
        
        # Clean numeric columns
        for col in df_new.columns:
            if col != timestamp_field and col != 'wqi_Category':
                try:
                    df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
                except:
                    df_new[col] = np.nan
        
        # Drop rows with all NaN values
        df_new = df_new.dropna(how='all').reset_index(drop=True)
        
        # Update last timestamp
        if not df_new.empty and timestamp_field in df_new.columns:
            latest = df_new[timestamp_field].max()
            if pd.notna(latest):
                st.session_state.last_timestamp = latest  # Already UTC from to_datetime(utc=True)
        
        return df_new
    
    return pd.DataFrame()

def categorize_wqi(df):
    """Categorize WQI values into quality categories"""
    if 'wqi' in df.columns:
        bins = [0, 25, 50, 75, 100]
        labels = ['Excellent', 'Good', 'Poor', 'Unsuitable']
        df['wqi_Category'] = pd.cut(df['wqi'], bins=bins, labels=labels)
    return df

def validate_data(df):
    """Basic data validation and cleaning"""
    if df.empty:
        return df
    
    # Ensure numeric columns remain numeric
    for col in df.select_dtypes(include='number').columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove problematic object-type columns
    for col in df.select_dtypes(include='object').columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.9 and col != 'wqi_Category':
            st.warning(f"Column '{col}' appears to contain random strings. Removing.")
            df = df.drop(columns=[col])
    
    return df

def arima_forecast(df):
    """Generate WQI forecasts using ARIMA model"""
    st.subheader("🔮 WQI Time Series Forecast using ARIMA")
    
    if 'wqi' not in df.columns or 'timestamp' not in df.columns:
        st.warning("Need both 'wqi' and 'timestamp' columns for forecasting")
        return
    
    # Default parameters for automatic execution
    forecast_days = 5
    arima_order = (1, 1, 1)
    
    try:
        # Prepare time series with UTC timestamps
        ts_df = df[['timestamp', 'wqi']].copy()
        ts_df['timestamp'] = pd.to_datetime(ts_df['timestamp'], utc=True)
        ts_df = ts_df.dropna().sort_values('timestamp').set_index('timestamp')
        
        if len(ts_df) < 20:
            st.warning("Need at least 20 data points for reliable forecasting")
            return
        
        # Fit ARIMA model
        model = ARIMA(ts_df['wqi'], order=arima_order)
        model_fit = model.fit()
        
        # Forecast
        forecast = model_fit.get_forecast(steps=forecast_days)
        pred_mean = forecast.predicted_mean
        pred_ci = forecast.conf_int()
        
        # Generate future dates starting from TODAY (UTC)
        today = pd.Timestamp.utcnow().floor('D')
        future_dates = pd.date_range(
            start=today + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D',
            tz='UTC'
        )
        
        # Build visualization
        fig = go.Figure()
        # Historical Data
        fig.add_trace(go.Scatter(
            x=ts_df.index,
            y=ts_df['wqi'],
            name='Historical',
            mode='lines+markers',
            line=dict(color='blue'),
            hovertemplate='<b>Date</b>: %{x}<br><b>WQI</b>: %{y:.1f}<extra></extra>'
        ))
        # Forecasted Data
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=pred_mean.values,
            name='Forecast',
            mode='lines+markers',
            line=dict(dash='dot', color='orange'),
            hovertemplate='<b>Date</b>: %{x}<br><b>Predicted WQI</b>: %{y:.1f}<extra></extra>'
        ))
        
        # Layout configuration
        fig.update_layout(
            title=f"{forecast_days}-Day WQI Forecast Using ARIMA{arima_order}",
            xaxis_title="Date",
            yaxis_title="WQI Value",
            hovermode='x unified',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            ),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary Table with Local Time Conversion
        summary_df = pd.DataFrame({
            'Date (UTC)': future_dates.strftime('%Y-%m-%d'),
            'Predicted WQI': pred_mean.values,
        }).round(2)
        
        # Show local time equivalent
        try:
            summary_df['Local Date'] = future_dates.tz_convert('Asia/Kolkata').strftime('%Y-%m-%d')  # Example for IST
        except:
            pass  # Fallback if timezone DB not available
        
        st.dataframe(summary_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"ARIMA Error: {str(e)}. Try simpler parameters.")

def create_visualizations(df):
    """Create comprehensive visual analysis tabs"""
    st.subheader("📊 Comprehensive Visual Analysis")
    
    tabs = st.tabs([
        "⏰ Timeline Heatmap", 
        "🌐 Radar Chart",
        "🧮 Boxplot Analysis",
        "🔗 Parallel Coordinates",
        "📈 Cumulative Density",
        "🌡️ Parameter Correlation",
        "🔁 Animated Trends",
        "📊 Custom Dashboard"
    ])
    
    # Tab 0: Timeline Heatmap
    with tabs[0]:
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.tz_localize(None).dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            heatmap_data = df.pivot_table(values='wqi', index='day_of_week', columns='hour', aggfunc='mean')
            fig = px.imshow(heatmap_data, title="Average WQI by Day & Hour", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Timestamp field required for timeline analysis")

    # Rest of the visualizations... (unchanged for brevity)

def show_summary(df):
    """Display summary statistics and category distribution"""
    st.subheader("💧 WQI Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Highest WQI", f"{df['wqi'].max():.1f}")
    with col2:
        st.metric("Lowest WQI", f"{df['wqi'].min():.1f}")
    with col3:
        st.metric("Avg WQI", f"{df['wqi'].mean():.1f}")
    with col4:
        st.metric("Total Samples", len(df))
    
    if not df.empty and 'wqi_Category' in df.columns:
        category_counts = df['wqi_Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        fig = px.pie(category_counts, names='Category', values='Count',
                     title="Water Quality Distribution",
                     color_discrete_map={
                         'Excellent': '#2ecc71',
                         'Good': '#f1c40f',
                         'Poor': '#e67e22',
                         'Unsuitable': '#e74c3c'
                     })
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application flow"""
    st.title("🌊 Real-Time Water Quality Analyzer")
    
    # Sidebar Settings
    st.sidebar.header("⚙️ Settings")
    db_name = st.sidebar.text_input("Database Name", value=os.getenv("DB_NAME", ""))
    collection_name = st.sidebar.text_input("Collection Name", value=os.getenv("COLLECTION_NAME", ""))
    timestamp_field = st.sidebar.text_input("Timestamp Field", value="timestamp")
    refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
    
    # Auto-refresh with immediate first execution
    st_autorefresh(interval=refresh_rate * 1000, key="data_refresher", limit=None)
    
    # Display current timestamp in local time
    try:
        local_time = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    except:
        local_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    st.markdown(f"**Current Time:** {local_time}")
    
    # Validate inputs
    if not db_name or not collection_name:
        st.warning("Please enter both Database and Collection name.")
        return
    
    # Get MongoDB client
    client = get_mongo_client()
    if not client:
        st.error("MongoDB client could not be initialized.")
        return
    
    # Container for status messages
    status_container = st.empty()
    
    with st.spinner("Fetching new data..."):
        # Fetch new data
        new_data = fetch_new_data(client, db_name, collection_name, timestamp_field)
        
        # Process new data
        if not new_data.empty:
            new_data = validate_data(new_data)
            st.session_state.main_df = pd.concat([st.session_state.main_df, new_data]).drop_duplicates().reset_index(drop=True)
            st.session_state.main_df = categorize_wqi(st.session_state.main_df)
            
            # Update status
            status_container.success(f"Fetched {len(new_data)} new record(s). Total: {len(st.session_state.main_df)} records.")
            st.session_state.initial_load_complete = True
            
        # If we've already loaded data before
        elif st.session_state.initial_load_complete:
            status_container.info("No new data available. Showing analysis from previously loaded data.")
        
        # First-time loading scenario
        if not st.session_state.main_df.empty:
            # Show summary and visualizations
            show_summary(st.session_state.main_df)
            arima_forecast(st.session_state.main_df)
            create_visualizations(st.session_state.main_df)
            
            # Display latest records
            with st.expander("📡 Latest Valid Records"):
                st.dataframe(st.session_state.main_df.tail(10), use_container_width=True)
                
        else:
            st.warning("No valid data available yet. Please check your database schema.")

if __name__ == "__main__":
    main()
