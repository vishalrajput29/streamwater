import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import pymongo
from streamlit_autorefresh import st_autorefresh
import ssl
from datetime import datetime, timedelta
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

load_dotenv()

# Session state initialization
if 'main_df' not in st.session_state:
    st.session_state.main_df = pd.DataFrame()
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = None
if 'initial_load_complete' not in st.session_state:
    st.session_state.initial_load_complete = False
if 'last_timestamp' not in st.session_state:
    st.session_state.last_timestamp = datetime.now()


def get_mongo_client():
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
    db = client[db_name]
    collection = db[collection_name]
    query = {}

    if st.session_state.last_fetch_time:
        query[timestamp_field] = {"$gt": st.session_state.last_timestamp}

    new_data = list(collection.find(query, {'_id': 0}))

    if new_data:
        df_new = pd.DataFrame(new_data)

        # Clean and coerce all numeric columns
        for col in df_new.columns:
            if col != timestamp_field and col != 'wqi_Category':
                try:
                    df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
                except Exception as e:
                    st.warning(f"Error coercing column '{col}': {str(e)}")
                    df_new[col] = np.nan

        # Drop rows with all NaN values
        df_new = df_new.dropna(how='all')

        if timestamp_field in df_new.columns:
            st.session_state.last_timestamp = df_new[timestamp_field].max() if timestamp_field in df_new.columns else datetime.now()

        return df_new

    return pd.DataFrame()


def categorize_wqi(df):
    if 'wqi' in df.columns:
        bins = [0, 25, 50, 75, 100]
        labels = ['Excellent', 'Good', 'Poor', 'Unsuitable']
        df['wqi_Category'] = pd.cut(df['wqi'], bins=bins, labels=labels)
    return df


def validate_data(df):
    """Basic data validation"""
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
    st.subheader("🔮 WQI Time Series Forecast using ARIMA")

    if 'wqi' not in df.columns or 'timestamp' not in df.columns:
        st.warning("Need both 'wqi' and 'timestamp' columns for forecasting")
        return

    col1, col2 = st.columns(2)
    with col1:
        forecast_days = st.slider("Forecast Days from Today", 1, 30, 5)
    with col2:
        arima_order = st.selectbox("ARIMA Order (p,d,q)",
                                   [(1, 1, 1), (2, 1, 1), (2, 1, 2), (3, 1, 2), (5, 1, 0)],
                                   format_func=lambda x: f"ARIMA{x}")

    try:
        # Prepare time series
        ts_df = df[['timestamp', 'wqi']].copy()
        ts_df['timestamp'] = pd.to_datetime(ts_df['timestamp'])
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

        # Generate future dates starting from TODAY
        today = pd.Timestamp.today().floor('D')
        future_dates = pd.date_range(
            start=today + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )

        # Create date-indexed predictions
        pred_mean = pd.Series(pred_mean.values, index=future_dates)
        pred_ci = pd.DataFrame(pred_ci.values, index=future_dates, columns=['Lower CI', 'Upper CI'])

        # --- Build Interactive Forecast Chart ---
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
            x=pred_mean.index,
            y=pred_mean.values,
            name='Forecast',
            mode='lines+markers',
            line=dict(dash='dot', color='orange'),
            hovertemplate='<b>Date</b>: %{x}<br><b>Predicted WQI</b>: %{y:.1f}<extra></extra>'
        ))

        # Confidence Interval
        fig.add_trace(go.Scatter(
            x=np.concatenate([pred_ci.index, pred_ci.index[::-1]]),
            y=np.concatenate([pred_ci.iloc[:, 0], pred_ci.iloc[:, 1][::-1]]),
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            hoverinfo='skip'
        ))

        # Add interactive buttons for timeframes
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

        # Summary Table with Dates
        summary_df = pd.DataFrame({
            'Date': pred_mean.index.strftime('%Y-%m-%d'),
            'Predicted WQI': pred_mean.values,
            'Lower Bound': pred_ci.iloc[:, 0],
            'Upper Bound': pred_ci.iloc[:, 1]
        }).round(2)

        st.dataframe(summary_df.set_index('Date'), use_container_width=True)

    except Exception as e:
        st.error(f"ARIMA Error: {str(e)}. Try simpler parameters.")


def create_visualizations(df):
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
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.day_name()
            heatmap_data = df.pivot_table(values='wqi', index='day_of_week', columns='hour', aggfunc='mean')
            fig = px.imshow(heatmap_data, title="Average WQI by Day & Hour", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Timestamp field required for timeline analysis")

    # Tab 1: Radar Chart
    with tabs[1]:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) > 1:
            categories = [col for col in numeric_cols if col != 'wqi']
            fig = go.Figure()
            for i in range(min(3, len(df))):
                fig.add_trace(go.Scatterpolar(r=df.iloc[i][categories].values.tolist(),
                                              theta=categories,
                                              fill='toself',
                                              name=f'Sample {i+1}'))
            fig.update_layout(title="Water Quality Parameters Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need multiple numeric parameters for radar chart")

    # Tab 2: Boxplot Analysis
    with tabs[2]:
        if len(df.select_dtypes(include='number').columns) > 1:
            param = st.selectbox("Select parameter for boxplot", df.select_dtypes(include='number').columns.tolist(), key='boxplot_param')
            fig = px.box(df, y=param, color='wqi_Category' if 'wqi_Category' in df.columns else None,
                         title=f"Distribution of {param} by Water Quality Category")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric data available for boxplot")

    # Tab 3: Parallel Coordinates
    with tabs[3]:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) > 1:
            fig = px.parallel_coordinates(df[numeric_cols], color="wqi", color_continuous_scale="Viridis",
                                         title="Parallel Coordinates Plot of Water Quality Parameters")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need multiple numeric parameters for parallel coordinates")

    # Tab 4: Cumulative Density
    with tabs[4]:
        if len(df.select_dtypes(include='number').columns) > 1:
            param = st.selectbox("Select parameter for CDF", df.select_dtypes(include='number').columns.tolist(), key='cdf_param')
            fig = px.ecdf(df, x=param, color='wqi_Category' if 'wqi_Category' in df.columns else None,
                          title=f"Cumulative Distribution of {param}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric data available for CDF")

    # Tab 5: Parameter Correlation
    with tabs[5]:
        corr_df = df.select_dtypes(include='number').corr()
        if not corr_df.empty and len(corr_df) > 1:
            fig = px.imshow(corr_df, text_auto=True, aspect="auto", title="Parameter Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need multiple numeric parameters for correlation analysis")

    # Tab 6: Animated Trends
    with tabs[6]:
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df_sorted = df.sort_values('datetime')
            param = st.selectbox("Select parameter for animation", df.select_dtypes(include='number').columns.tolist(), key='anim_param')
            fig = px.scatter(df_sorted, x='datetime', y=param,
                             animation_frame=df_sorted['datetime'].dt.strftime('%Y-%m-%d'),
                             range_y=[df_sorted[param].min() * 0.9, df_sorted[param].max() * 1.1],
                             title=f"Animated Trend of {param} Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Timestamp field required for animated trends")

    # Tab 7: Custom Dashboard
    with tabs[7]:
        st.markdown("### 📊 Custom Analysis Dashboard")
        viz_type = st.selectbox("Choose Visualization Type", ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram"])
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()

        if viz_type == "Scatter Plot":
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X-axis", numeric_cols, key='scatter_x')
                y_col = st.selectbox("Y-axis", numeric_cols, key='scatter_y')
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least two numeric columns for scatter plot")

        elif viz_type == "Line Chart":
            if 'timestamp' in df.columns:
                y_col = st.selectbox("Y-axis", numeric_cols, key='line_y')
                fig = px.line(df, x='timestamp', y=y_col, title=f"Trend of {y_col} Over Time")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need timestamp field for line charts")

        elif viz_type == "Bar Chart":
            if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
                cat_col = st.selectbox("Category", cat_cols, key='bar_cat')
                val_col = st.selectbox("Value", numeric_cols, key='bar_val')
                fig = px.bar(df.groupby(cat_col)[val_col].mean().reset_index(),
                             x=cat_col, y=val_col, title=f"Average {val_col} by {cat_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least one categorical and one numeric column for bar charts")

        elif viz_type == "Histogram":
            if len(numeric_cols) >= 1:
                hist_col = st.selectbox("Column to Analyze", numeric_cols, key='hist_col')
                fig = px.histogram(df, x=hist_col, nbins=30, marginal="rug", title=f"Distribution of {hist_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need numeric data for histograms")


def show_summary(df):
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
    st.title("🌊 Real-Time Water Quality Analyzer")

    st.sidebar.header("⚙️ Settings")
    db_name = st.sidebar.text_input("Database Name", value=os.getenv("DB_NAME", ""))
    collection_name = st.sidebar.text_input("Collection Name", value=os.getenv("COLLECTION_NAME", ""))
    timestamp_field = st.sidebar.text_input("Timestamp Field", value="timestamp")
    refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)

    st_autorefresh(interval=refresh_rate * 1000, key="data_refresher")
    st.markdown(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not db_name or not collection_name:
        st.warning("Please enter both Database and Collection name.")
        return

    client = get_mongo_client()
    if not client:
        st.error("MongoDB client could not be initialized.")
        return

    with st.spinner("Fetching new data..."):
        new_data = fetch_new_data(client, db_name, collection_name, timestamp_field)

        if not new_data.empty:
            new_data = validate_data(new_data)
            st.session_state.main_df = pd.concat([st.session_state.main_df, new_data]).drop_duplicates().reset_index(drop=True)
            st.session_state.main_df = categorize_wqi(st.session_state.main_df)
            st.success(f"Fetched {len(new_data)} new record(s). Total: {len(st.session_state.main_df)} records.")
            st.session_state.initial_load_complete = True

        elif st.session_state.initial_load_complete:
            st.info("No new data available. Showing analysis from previously loaded data.")

        if not st.session_state.main_df.empty:
            show_summary(st.session_state.main_df)
            arima_forecast(st.session_state.main_df)
            create_visualizations(st.session_state.main_df)
        else:
            st.warning("No valid data available yet. Please check your database schema.")

    if not st.session_state.main_df.empty:
        with st.expander("📡 Latest Valid Records"):
            st.dataframe(st.session_state.main_df.tail(10), use_container_width=True)


if __name__ == "__main__":
    main()
