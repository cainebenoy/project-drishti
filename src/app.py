import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import numpy as np

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
</style>
""", unsafe_allow_html=True)

# --- DATA LOADER ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/final_scored_data.csv")
    except FileNotFoundError:
        st.error("❌ Data not found. Please run engine.py first.")
        return pd.DataFrame()

    # --- COORDINATE SIMULATION (HACK FOR DEMO) ---
    state_coords = {
        "Delhi": [28.7041, 77.1025],
        "Maharashtra": [19.7515, 75.7139],
        "Uttar Pradesh": [26.8467, 80.9462],
        "Karnataka": [15.3173, 75.7139],
        "Bihar": [25.0961, 85.3131],
        "Tamil Nadu": [11.1271, 78.6569],
        "West Bengal": [22.9868, 87.8550],
        "Gujarat": [22.2587, 71.1924],
        "Rajasthan": [27.0238, 74.2179],
    }
    
    def get_lat(row):
        base = state_coords.get(row['state'], [20.5937, 78.9629])
        jitter = (int(row['pincode']) % 100) / 100.0 
        return base[0] + (jitter * 0.1)

    def get_lon(row):
        base = state_coords.get(row['state'], [20.5937, 78.9629])
        jitter = (int(row['pincode']) % 200) / 100.0 
        return base[1] + (jitter * 0.1)

    df['lat'] = df.apply(get_lat, axis=1)
    df['lon'] = df.apply(get_lon, axis=1)
    
    return df

df = load_data()

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
        ["All", "CRITICAL Only", "Low Risk"]
    )

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

# --- MAIN DASHBOARD ---
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

st.markdown("---")

# Row 1: The Map & The Analysis
m_col1, m_col2 = st.columns([2, 1])

with m_col1:
    st.subheader("📍 Geospatial Anomaly Distribution")
    if not df_filtered.empty:
        map_center = [df_filtered['lat'].mean(), df_filtered['lon'].mean()]
        m = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB dark_matter')

        plot_data = df_filtered.head(1000)
        
        for _, row in plot_data.iterrows():
            color = "#FF0000" if row['risk_status'] == 'CRITICAL' else "#00FFFF"
            radius = 6 if row['risk_status'] == 'CRITICAL' else 2
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Pincode: {row['pincode']}\nRisk: {row['primary_risk_factor']}"
            ).add_to(m)

        st_folium(m, width=None, height=400)
    else:
        st.info("No data to display.")

with m_col2:
    st.subheader("📊 Risk Factor Breakdown")
    if not df_filtered.empty:
        critical_data = df_filtered[df_filtered['risk_status'] == 'CRITICAL']
        if not critical_data.empty:
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
            * **Velocity Index:** Updates happening too fast (Batch Processing).
            * **Ghost Ratio:** Address changes without biometric checks.
            * **Spike Ratio:** Sudden surge in new adults.
            """)
        else:
            st.success("No Critical Risks in this view.")

# Row 2: The Hit List & XAI
st.markdown("---")
st.subheader("🚨 Priority Audit List (Select to Inspect)")

if not df_filtered.empty:
    risky_table = df_filtered[df_filtered['risk_status'] == 'CRITICAL'][
        ['pincode', 'state', 'district', 'velocity_index', 'adult_spike_ratio', 'ghost_ratio', 'primary_risk_factor']
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
        
        # Explainable AI Section (XAI)
        if len(event.selection['rows']) > 0:
            selected_index = event.selection['rows'][0]
            selected_row = risky_table.iloc[selected_index]
            
            st.markdown("### 🔍 XAI Inspector: Why was this Pincode Flagged?")
            
            # Create a localized bar chart for this specific row
            xai_data = pd.DataFrame({
                'Risk Factor': ['Velocity', 'Ghost Ratio', 'Adult Spike'],
                'Value': [selected_row['velocity_index'], selected_row['ghost_ratio']*1000, selected_row['adult_spike_ratio']*1000] 
                # Scaled for visualization comparison
            })
            
            fig_bar = px.bar(
                xai_data, 
                x='Risk Factor', 
                y='Value', 
                color='Risk Factor',
                title=f"Anomaly Profile: Pincode {selected_row['pincode']}"
            )
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_bar, use_container_width=True)
            
    else:
        st.success("Region is Secure.")