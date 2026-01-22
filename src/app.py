import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import numpy as np
import os
import sys

# --- IMPORT PDF GENERATOR ---
# We force the path to look in the current directory
sys.path.append(os.path.dirname(__file__))

PDF_MODULE_AVAILABLE = False
try:
    from reporter import generate_mission_pdf
    PDF_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"PDF Generator Error: {e}")
    # We will show a warning in the UI later if this is False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project Drishti | UIDAI Integrity",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0E1117; border-right: 1px solid #262730; }
    .metric-card { background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-left: 5px solid #00FFFF; }
    .risk-card { background-color: #2D0F0F; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B; }
    h1, h2, h3 { color: #E0E0E0; }
    div[data-testid="stMetricValue"] { color: #00FFFF; }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADER ---
@st.cache_data
def load_data_engine():
    # Load Geocoded Data (Best)
    if os.path.exists("data/final_scored_data_geocoded.csv"):
        try:
            return pd.read_csv("data/final_scored_data_geocoded.csv"), "High-Precision Satellite Data Loaded", "🛰️"
        except: pass
    
    # Load Standard Data (Good)
    if os.path.exists("data/final_scored_data.csv"):
        try:
            return pd.read_csv("data/final_scored_data.csv"), "Standard Dataset Loaded", "ℹ️"
        except: pass
        
    return pd.DataFrame(), "No Data Found. Run engine.py", "❌"

def process_coordinates(df):
    if df.empty: return df
    
    # Defaults
    for col in ['lat', 'lon']: 
        if col not in df.columns: df[col] = 0.0
    if 'geo_accuracy' not in df.columns: df['geo_accuracy'] = 'Low'

    state_coords = {
        "Delhi": [28.7041, 77.1025], "Maharashtra": [19.7515, 75.7139],
        "Uttar Pradesh": [26.8467, 80.9462], "Karnataka": [15.3173, 75.7139],
        "Bihar": [25.0961, 85.3131], "Tamil Nadu": [11.1271, 78.6569],
        "West Bengal": [22.9868, 87.8550], "Gujarat": [22.2587, 71.1924],
        "Rajasthan": [27.0238, 74.2179],
    }
    
    def fill_coords(row):
        if not pd.isna(row['lat']) and row['lat'] != 0.0:
            return row['lat'], row['lon']
        base = state_coords.get(row['state'], [20.5937, 78.9629])
        jitter = (int(row['pincode']) % 100) / 100.0 
        return base[0] + (jitter * 0.1), base[1] + (jitter * 0.2)

    coords = df.apply(fill_coords, axis=1)
    df['lat'] = [x[0] for x in coords]
    df['lon'] = [x[1] for x in coords]
    return df

# --- APP EXECUTION ---
raw_df, msg, icon = load_data_engine()
if not raw_df.empty and msg: st.toast(msg, icon=icon)
df = process_coordinates(raw_df)

# Sidebar
st.sidebar.title("👁️ Drishti Controls")
st.sidebar.markdown("---")

if not df.empty:
    state_filter = st.sidebar.multiselect("Filter by State", df['state'].unique(), df['state'].unique()[:3])
    risk_filter = st.sidebar.radio("Threat Level", ["All", "CRITICAL Only", "Low Risk"], index=1)
    
    df_filtered = df.copy()
    if state_filter: df_filtered = df_filtered[df_filtered['state'].isin(state_filter)]
    if risk_filter == "CRITICAL Only": df_filtered = df_filtered[df_filtered['risk_status'] == 'CRITICAL']
    elif risk_filter == "Low Risk": df_filtered = df_filtered[df_filtered['risk_status'] == 'Low']
    
    st.sidebar.download_button("📥 Download Audit CSV", df_filtered.to_csv(index=False).encode('utf-8'), "drishti_report.csv")
else:
    df_filtered = pd.DataFrame()

# Main Layout
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Project Drishti | Integrity Surveillance")
    if not PDF_MODULE_AVAILABLE:
        st.warning("⚠️ PDF Module not loaded. Install reportlab: `pip install reportlab`")
with col2:
    if not df_filtered.empty:
        risk_count = len(df_filtered[df_filtered['risk_status'] == 'CRITICAL'])
        st.markdown(f"""<div class="risk-card"><h3>Active Threats</h3><h1 style="color: #FF4B4B; margin:0;">{risk_count}</h1></div>""", unsafe_allow_html=True)

st.markdown("---")

# Visuals
m_col1, m_col2 = st.columns([2, 1])
with m_col1:
    st.subheader("📍 Geospatial View")
    if not df_filtered.empty:
        center = [df_filtered['lat'].mean(), df_filtered['lon'].mean()] if not df_filtered.empty else [20, 78]
        m = folium.Map(location=center, zoom_start=5, tiles='CartoDB dark_matter')
        
        # Limit markers
        for _, row in df_filtered.head(500).iterrows():
            color = "#FF0000" if row['risk_status'] == 'CRITICAL' else "#00FFFF"
            folium.CircleMarker([row['lat'], row['lon']], radius=4, color=color, fill=True, fill_color=color).add_to(m)
        st_folium(m, height=400, width=None)

with m_col2:
    st.subheader("📊 Anomaly Breakdown")
    if not df_filtered.empty:
        crit = df_filtered[df_filtered['risk_status'] == 'CRITICAL']
        if not crit.empty:
            fig = px.pie(crit, names='primary_risk_factor', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)

# Hit List & Actions
st.markdown("---")
st.subheader("🚨 Priority Audit List (Select Row to Generate PDF)")

if not df_filtered.empty:
    risky = df_filtered[df_filtered['risk_status'] == 'CRITICAL'].sort_values('velocity_index', ascending=False)
    
    event = st.dataframe(
        risky[['pincode', 'state', 'district', 'velocity_index', 'primary_risk_factor']],
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun"
    )

    # ACTION SECTION
    if len(event.selection['rows']) > 0:
        idx = event.selection['rows'][0]
        row = risky.iloc[idx]
        
        st.markdown(f"### ⚡ Action Console: {row['district']} ({row['pincode']})")
        
        ac1, ac2 = st.columns(2)
        with ac1:
            st.info(f"Primary Risk: {row['primary_risk_factor']}")
            if st.button("📡 Push Alert to SITAA"):
                st.toast("✅ Alert Sent to Mainframe", icon="🔗")
        
        with ac2:
            if PDF_MODULE_AVAILABLE:
                # PDF Generation
                try:
                    pdf_data = generate_mission_pdf(row)
                    st.download_button(
                        label="📄 DOWNLOAD MISSION ORDER (PDF)",
                        data=pdf_data,
                        file_name=f"MISSION_{row['pincode']}.pdf",
                        mime="application/pdf",
                        type="primary"  # Red button
                    )
                except Exception as e:
                    st.error(f"PDF Error: {e}")
            else:
                st.error("PDF Module Missing. Check logs.")