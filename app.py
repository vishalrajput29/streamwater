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
        st.session_state.last_timestamp = df_new[timestamp_field].max() if timestamp_field in df_new.columns else datetime.now()
        return df_new
    return pd.DataFrame()

def create_interactive_charts(df):
    st.subheader("📊 Advanced Visualizations")

    # Tabbed interface
    tab1, tab2, tab3 = st.tabs(["📈 Parameter Trends", "🧮 wqi Distribution", "🔍 Multi-Parameter Analysis"])

    with tab1:
        numeric_cols = [col for col in df.select_dtypes(include='number').columns if col != 'wqi']
        if numeric_cols:
            selected_col = st.selectbox("Select Parameter to Track Over Time", numeric_cols, key='trend')
            fig = px.line(df, x='timestamp', y=selected_col, color='wqi_Category',
                         title=f"{selected_col} Trend with wqi Category")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        wqi_range = [df['wqi'].min(), df['wqi'].max()]
        bins = [0, 25, 50, 75, 100]
        labels = ['Excellent (0-25)', 'Good (25-50)', 'Poor (50-75)', 'Unsuitable (>75)']
        
        df['wqi_Category'] = pd.cut(df['wqi'], bins=bins, labels=labels)
        
        fig = px.histogram(df, x='wqi', color='wqi_Category', marginal="box",
                          title="Water Quality Index (wqi) Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        params = [col for col in df.columns if col not in ['timestamp', 'wqi', 'wqi_Category']]
        selected_params = st.multiselect("Select Parameters for Correlation", params[:5])
        
        if len(selected_params) >= 2:
            fig = px.scatter_matrix(df, dimensions=selected_params + ['wqi'],
                                  color='wqi_Category', opacity=0.6,
                                  title="Multi-Parameter Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

def show_wqi_summary(df):
    st.subheader("💧 wqi Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Highest wqi", f"{df['wqi'].max():.1f}")
    with col2:
        st.metric("Lowest wqi", f"{df['wqi'].min():.1f}")
    with col3:
        st.metric("Avg wqi", f"{df['wqi'].mean():.1f}")
    with col4:
        st.metric("Total Samples", len(df))

    # Categorize wqi
    bins = [0, 25, 50, 75, 100]
    labels = ['Excellent', 'Good', 'Poor', 'Unsuitable']
    df['wqi_Category'] = pd.cut(df['wqi'], bins=bins, labels=labels)
    
    category_counts = df['wqi_Category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    
    fig = px.bar(category_counts, x='Category', y='Count', 
                title="Water Quality Categories", color='Category',
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
            # Ensure wqi field exists
            if 'wqi' not in new_data.columns:
                st.error("No 'wqi' column found in your data. Please verify your database schema.")
                return
                
            # Merge with existing data
            st.session_state.main_df = pd.concat([st.session_state.main_df, new_data]).drop_duplicates().reset_index(drop=True)
            st.success(f"Fetched {len(new_data)} new record(s). Total: {len(st.session_state.main_df)} records.")
            st.session_state.initial_load_complete = True
        elif st.session_state.initial_load_complete:
            st.info("No new data available. Showing analysis from previously loaded data.")

        if not st.session_state.main_df.empty:
            # Show summary statistics
            show_wqi_summary(st.session_state.main_df)
            
            # Create interactive visualizations
            create_interactive_charts(st.session_state.main_df)
        else:
            st.warning("No data available yet. Please ensure your database contains 'wqi' values.")

    # Optional: Display recent data
    if not st.session_state.main_df.empty:
        st.subheader("📡 Latest Records")
        st.dataframe(st.session_state.main_df.tail(10), use_container_width=True)

if __name__ == "__main__":
    main()