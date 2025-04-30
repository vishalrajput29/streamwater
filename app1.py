import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import pymongo
from streamlit_autorefresh import st_autorefresh
import ssl
from datetime import datetime
import numpy as np

load_dotenv()

# Session state initialization
if 'main_df' not in st.session_state:
    st.session_state.main_df = pd.DataFrame()
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = None
if 'initial_load_complete' not in st.session_state:
    st.session_state.initial_load_complete = False

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

def create_visualizations(df):
    st.subheader("📊 Comprehensive Visual Analysis")
    
    # Tabbed interface with 8 visualizations
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
    
    try:
        with tabs[0]:  # Heatmap over time
            numeric_cols = [col for col in df.select_dtypes(include='number').columns if col != 'wqi']
            if numeric_cols:
                selected_col = st.selectbox("Select Column for Heatmap", numeric_cols, key='heatmap')
                fig = px.density_heatmap(df, x='timestamp', y=selected_col, z='wqi',
                                       title=f"{selected_col} vs Time with WQI Intensity")
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Heatmap visualization error: {str(e)}")

    try:
        with tabs[1]:  # Radar Chart
            params = [col for col in df.columns if col not in ['timestamp', 'wqi', 'wqi_Category']]
            if len(params) >= 3:
                categories = df['wqi_Category'].unique()
                selected_category = st.selectbox("Select Category for Radar", categories, key='radar_cat')
                
                filtered = df[df['wqi_Category'] == selected_category]
                values = filtered[params].mean().values.tolist()
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=values + [values[0]],
                    theta=params + [params[0]],
                    fill='toself',
                    name='Average Values'
                ))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                                title=f"Radar View for {selected_category} Quality Water")
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Radar chart visualization error: {str(e)}")

    try:
        with tabs[2]:  # Boxplot by category
            fig = px.box(df, x='wqi_Category', y='wqi', color='wqi_Category',
                        title="WQI Distribution by Category",
                        category_orders={'wqi_Category': ['Excellent', 'Good', 'Poor', 'Unsuitable']},
                        color_discrete_map={
                            'Excellent': '#2ecc71',
                            'Good': '#f1c40f',
                            'Poor': '#e67e22',
                            'Unsuitable': '#e74c3c'
                        })
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Boxplot visualization error: {str(e)}")

    try:
        with tabs[3]:  # Parallel coordinates
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if len(numeric_cols) >= 4:
                fig = px.parallel_coordinates(df, 
                                            dimensions=numeric_cols[:4],
                                            color='wqi',
                                            color_continuous_scale=px.colors.sequential.Viridis,
                                            title="Multi-Parameter Parallel View")
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Parallel coordinates visualization error: {str(e)}")

    try:
        with tabs[4]:  # CDF Plot
            fig = px.ecdf(df, x='wqi', color='wqi_Category',
                        title="Cumulative Distribution of WQI Values")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"CDF plot visualization error: {str(e)}")

    try:
        with tabs[5]:  # Correlation heatmap
            numeric_cols = df.select_dtypes(include='number').drop(columns=['wqi']).columns.tolist()
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols + ['wqi']].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title="Parameter Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Correlation heatmap visualization error: {str(e)}")

    try:
        with tabs[6]:  # Animated time series
            fig = px.scatter(df, x='timestamp', y='wqi', animation_frame='wqi_Category',
                            range_y=[df['wqi'].min()-10, df['wqi'].max()+10],
                            title="WQI Evolution Over Time (Animated)")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Animated trends visualization error: {str(e)}")

    try:
        with tabs[7]:  # Custom dashboard grid
            col1, col2 = st.columns(2)
            
            with col1:
                valid_params = df.columns.drop(['timestamp', 'wqi_Category'])
                param1 = st.selectbox("First Parameter", valid_params, key='grid1')
                fig = px.area(df, x='timestamp', y=param1, color='wqi_Category',
                            title=f"{param1} Over Time")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                param2 = st.selectbox("Second Parameter", valid_params, key='grid2')
                fig = px.bar(df.tail(10), x='timestamp', y=param2, color='wqi_Category',
                            title=f"{param2} Comparison")
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Custom dashboard visualization error: {str(e)}")

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
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Settings")
    db_name = st.sidebar.text_input("Database Name", value=os.getenv("DB_NAME", ""))
    collection_name = st.sidebar.text_input("Collection Name", value=os.getenv("COLLECTION_NAME", ""))
    timestamp_field = st.sidebar.text_input("Timestamp Field", value="timestamp")
    refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
    
    # Auto-refresh logic
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
            # Validate data structure
            new_data = validate_data(new_data)
            
            # Merge and categorize
            st.session_state.main_df = pd.concat([st.session_state.main_df, new_data]).drop_duplicates().reset_index(drop=True)
            st.session_state.main_df = categorize_wqi(st.session_state.main_df)
            
            st.success(f"Fetched {len(new_data)} new record(s). Total: {len(st.session_state.main_df)} records.")
            st.session_state.initial_load_complete = True
        elif st.session_state.initial_load_complete:
            st.info("No new data available. Showing analysis from previously loaded data.")

        if not st.session_state.main_df.empty:
            # Show summary statistics
            show_summary(st.session_state.main_df)
            
            # Create visualizations
            create_visualizations(st.session_state.main_df)
        else:
            st.warning("No valid data available yet. Please check your database schema.")

    # Optional: Display recent data
    if not st.session_state.main_df.empty:
        with st.expander("📡 Latest Valid Records"):
            st.dataframe(st.session_state.main_df.tail(10), use_container_width=True)

if __name__ == "__main__":
    main()