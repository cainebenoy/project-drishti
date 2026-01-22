import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import numpy as np
import os
import sys

# --- SETUP & IMPORTS ---
# Add the src directory to path so we can import local modules
sys.path.append(os.path.dirname(__file__))

# Import the PDF Generator (Reporter)
try:
    from reporter import generate_mission_pdf
    PDF_MODULE_AVAILABLE = True
except ImportError:
    PDF_MODULE_AVAILABLE = False

# Import the Forecaster
try:
    from forecaster import generate_forecast
    FORECAST_MODULE_AVAILABLE = True
except ImportError:
    FORECAST_MODULE_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project Drishti | UIDAI Integrity",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Dark Mode" Vibe
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid #262730;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00FFFF;
    }
    .risk-card {
        background-color: #2D0F0F;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
    }
    h1, h2, h3 { color: #E0E0E0; }
    div[data-testid="stMetricValue"] { color: #00FFFF; }
    
    /* Make tabs visible in dark mode */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0E1117;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E1E1E;
        color: #00FFFF;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADER (Pure Data Logic) ---
@st.cache_data
def load_data_engine():
    # Priority: Load Geocoded data if available
    if os.path.exists("data/final_scored_data_geocoded.csv"):
        try:
            df = pd.read_csv("data/final_scored_data_geocoded.csv")
            return df, "High-Precision Satellite Data Loaded", "🛰️"
        except Exception:
            pass # Fallback to standard
            
    # Load Standard Data
    if os.path.exists("data/final_scored_data.csv"):
        try:
            df = pd.read_csv("data/final_scored_data.csv")
            return df, "Using Standard Dataset (Run geocoder.py for precision)", "ℹ️"
        except Exception:
             pass

    return pd.DataFrame(), "Data not found. Please run engine.py first.", "❌"

def process_coordinates(df):
    if df.empty: return df

    # Ensure required columns exist
    for col in ['lat', 'lon']:
        if col not in df.columns: df[col] = 0.0
    if 'geo_accuracy' not in df.columns: df['geo_accuracy'] = 'Low'

    # Fallback Simulation for Low-Accuracy rows (Safe Zones)
    state_coords = {
        "Delhi": [28.7041, 77.1025], "Maharashtra": [19.7515, 75.7139],
        "Uttar Pradesh": [26.8467, 80.9462], "Karnataka": [15.3173, 75.7139],
        "Bihar": [25.0961, 85.3131], "Tamil Nadu": [11.1271, 78.6569],
        "West Bengal": [22.9868, 87.8550], "Gujarat": [22.2587, 71.1924],
        "Rajasthan": [27.0238, 74.2179],
    }
    
    def fill_coords(row):
        # Keep real coords if they exist
        if not pd.isna(row['lat']) and row['lat'] != 0.0:
            return row['lat'], row['lon']
        
        # Simulate
        base = state_coords.get(row['state'], [20.5937, 78.9629])
        jitter_lat = (int(row['pincode']) % 100) / 100.0 
        jitter_lon = (int(row['pincode']) % 200) / 100.0 
        return base[0] + (jitter_lat * 0.1), base[1] + (jitter_lon * 0.1)

    # Apply filling
    coords = df.apply(fill_coords, axis=1)
    df['lat'] = [x[0] for x in coords]
    df['lon'] = [x[1] for x in coords]
    
    return df

# --- APP EXECUTION STARTS HERE ---
raw_df, msg, icon = load_data_engine()

# Display Status Toast
if not raw_df.empty and msg:
    st.toast(msg, icon=icon)
elif msg:
    st.error(msg)

df = process_coordinates(raw_df)

# --- SIDEBAR FILTERS ---
st.sidebar.title("👁️ Drishti Controls")
st.sidebar.markdown("---")

if not df.empty:
    state_filter = st.sidebar.multiselect(
        "Filter by State",
        options=df['state'].unique(),
        default=df['state'].unique()[:3]
    )

    risk_filter = st.sidebar.radio(
        "Threat Level",
        ["All", "CRITICAL Only", "Low Risk"],
        index=1 # Default to Critical
    )

    # Apply Filters
    df_filtered = df.copy()
    if state_filter:
        df_filtered = df_filtered[df_filtered['state'].isin(state_filter)]

    if risk_filter == "CRITICAL Only":
        df_filtered = df_filtered[df_filtered['risk_status'] == 'CRITICAL']
    elif risk_filter == "Low Risk":
        df_filtered = df_filtered[df_filtered['risk_status'] == 'Low']
    
    st.sidebar.markdown("---")
    st.sidebar.info("Model: Isolation Forest (Unsupervised)\nExplainability: SHAP")
    
    # Download Button
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        "📥 Download Audit Report",
        csv,
        "drishti_audit_report.csv",
        "text/csv",
        key='download-csv'
    )

else:
    df_filtered = pd.DataFrame()

# --- MAIN HEADER ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Project Drishti | Integrity Surveillance")
    st.markdown("**Real-time Anomaly Detection Engine for Aadhaar Ecosystem**")
with col2:
    if not df_filtered.empty:
        risk_count = len(df_filtered[df_filtered['risk_status'] == 'CRITICAL'])
        st.markdown(f"""
        <div class="risk-card">
            <h3>Active Threats</h3>
            <h1 style="color: #FF4B4B; margin:0;">{risk_count}</h1>
        </div>
        """, unsafe_allow_html=True)

# --- TABS FOR DIFFERENT VIEWS ---
tab1, tab2 = st.tabs(["🚨 Surveillance Map", "📈 Predictive Analytics"])

# --- TAB 1: SURVEILLANCE MAP ---
with tab1:
    m_col1, m_col2 = st.columns([2, 1])

    with m_col1:
        st.subheader("📍 Geospatial Anomaly Distribution")
        
        if not df_filtered.empty:
            # Map Centering Logic
            if risk_filter == "CRITICAL Only" and not df_filtered.empty:
                 map_center = [df_filtered['lat'].mean(), df_filtered['lon'].mean()]
                 zoom = 6
            else:
                 map_center = [20.5937, 78.9629] # Center of India
                 zoom = 5
                 
            m = folium.Map(location=map_center, zoom_start=zoom, tiles='CartoDB dark_matter')

            # Limit points for performance
            plot_data = df_filtered.head(1000)
            
            for _, row in plot_data.iterrows():
                is_critical = row['risk_status'] == 'CRITICAL'
                color = "#FF0000" if is_critical else "#00FFFF"
                radius = 6 if is_critical else 2
                
                # Label Accuracy Tag
                accuracy_tag = "🛰️ GPS Verified" if row.get('geo_accuracy') == 'High' else "⚠️ Approx Loc"
                
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    popup=folium.Popup(f"<b>Pincode: {row['pincode']}</b><br>Risk: {row['primary_risk_factor']}<br><small>{accuracy_tag}</small>", max_width=200)
                ).add_to(m)

            st_folium(m, width=None, height=400)
        else:
            st.info("No data to display with current filters.")

    with m_col2:
        st.subheader("📊 Risk Factor Breakdown")
        if not df_filtered.empty:
            critical_data = df_filtered[df_filtered['risk_status'] == 'CRITICAL']
            if not critical_data.empty:
                # Pie chart
                fig = px.pie(
                    critical_data, 
                    names='primary_risk_factor', 
                    title='Dominant Anomalies',
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    hole=0.4
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Key:**
                * **Velocity Index:** Updates happening too fast.
                * **Ghost Ratio:** Address changes w/o biometric checks.
                * **Spike Ratio:** Sudden surge in new adults.
                """)
            else:
                st.success("No Critical Risks in this view.")

    # Hit List & Actions
    st.markdown("---")
    st.subheader("🚨 Priority Audit List (Select Row to Generate PDF)")

    if not df_filtered.empty:
        risky_table = df_filtered[df_filtered['risk_status'] == 'CRITICAL'][
            ['pincode', 'state', 'district', 'velocity_index', 'adult_spike_ratio', 'ghost_ratio', 'primary_risk_factor', 'lat', 'lon']
        ].sort_values(by='velocity_index', ascending=False)
        
        if not risky_table.empty:
            # Selection Logic
            event = st.dataframe(
                risky_table, 
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "velocity_index": st.column_config.ProgressColumn(
                        "Velocity Load", format="%d", min_value=0, max_value=int(df['velocity_index'].max())),
                    "ghost_ratio": st.column_config.NumberColumn("Ghost Ratio", format="%.2f"),
                    "adult_spike_ratio": st.column_config.NumberColumn("Adult Spike", format="%.2f"),
                }
            )
            
            # Action Console (Only shows if row selected)
            if len(event.selection['rows']) > 0:
                selected_index = event.selection['rows'][0]
                selected_row = risky_table.iloc[selected_index]
                
                st.markdown(f"### ⚡ Action Console: {selected_row['district']} ({selected_row['pincode']})")
                
                # Layout: Left for Info, Right for PDF
                ac1, ac2 = st.columns(2)
                
                with ac1:
                    st.info(f"Anomaly Detected: {selected_row['primary_risk_factor']}")
                    if st.button("📡 Push Alert to SITAA"):
                        st.toast(f"✅ Alert Ticket #{np.random.randint(10000,99999)} created in SITAA Mainframe.", icon="🔗")
                
                with ac2:
                    if PDF_MODULE_AVAILABLE:
                        # PDF Generation
                        try:
                            pdf_data = generate_mission_pdf(selected_row)
                            st.download_button(
                                label="📄 DOWNLOAD MISSION ORDER (PDF)",
                                data=pdf_data,
                                file_name=f"MISSION_{selected_row['pincode']}.pdf",
                                mime="application/pdf",
                                type="primary"  # Red button
                            )
                        except Exception as e:
                            st.error(f"PDF Error: {e}")
                    else:
                        st.warning("⚠️ PDF Module unavailable. Install reportlab.")
                
        else:
            st.success("Region is Secure. No anomalies detected.")
    else:
        st.info("Awaiting Data...")

# --- TAB 2: PREDICTIVE ANALYTICS ---
with tab2:
    st.subheader("🔮 Demand Forecasting (Next 30 Days)")
    st.markdown("Prophet AI model prediction for biometric update volume. Used for resource allocation.")
    
    if FORECAST_MODULE_AVAILABLE:
        try:
            fig_forecast = generate_forecast(days=30)
            if fig_forecast:
                st.plotly_chart(fig_forecast, use_container_width=True)
            else:
                st.warning("Insufficient data points for forecasting. Need at least 5 days of history.")
        except Exception as e:
            st.error(f"Forecasting Error: {e}")
    else:
        st.error("Forecaster module missing. Please install `prophet`.")