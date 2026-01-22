import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import os
import streamlit as st

@st.cache_resource(show_spinner=False)
def get_forecast_model():
    """
    Loads data and trains the Prophet model ONCE. 
    Returns the original dataframe and the forecast dataframe.
    Uses @st.cache_resource to prevent re-training on every interaction.
    """
    data_path = "data/daily_timeseries.csv"
    
    # 1. Path Check
    if not os.path.exists(data_path):
        return None

    try:
        # 2. Load Data
        df = pd.read_csv(data_path)
        
        # 3. Data Cleaning
        # Prophet strictly requires columns named 'ds' (date) and 'y' (target)
        if 'ds' not in df.columns or 'y' not in df.columns:
            # Try to infer if columns are misnamed or just grab by position
            # Assuming first col is date, last is value if names don't match
            if len(df.columns) >= 2:
                 df = df.rename(columns={df.columns[0]: 'ds', df.columns[-1]: 'y'})
            else:
                return None
            
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df = df.dropna(subset=['ds', 'y'])
        
        # Sanity Check: Need minimal points to train a time-series
        if len(df) < 5: 
            return None 

        # 4. Train Model
        # Using lightweight parameters for speed in a hackathon demo
        m = Prophet(
            daily_seasonality=False,
            yearly_seasonality=False,
            weekly_seasonality=True,
            changepoint_prior_scale=0.1
        )
        m.fit(df)
        
        # 5. Forecast 30 days into future
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        
        return df, forecast
        
    except Exception as e:
        print(f"Forecasting Engine Error: {e}")
        return None

def generate_forecast_plot():
    """
    Generates the Plotly figure using the cached model results.
    """
    result = get_forecast_model()
    
    if not result:
        return None
    
    df, forecast = result
    
    # --- VISUALIZATION ---
    fig = go.Figure()

    # A. Historical Data (Cyan Line)
    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['y'],
        mode='lines', name='Historical Volume',
        line=dict(color='#00FFFF', width=1.5),
        opacity=0.7
    ))

    # B. AI Forecast (Red Dotted Line)
    pred_data = forecast.tail(30)
    fig.add_trace(go.Scatter(
        x=pred_data['ds'], y=pred_data['yhat'],
        mode='lines', name='AI Forecast (30 Days)',
        line=dict(color='#FF4B4B', width=2.5, dash='dot')
    ))

    # C. Confidence Interval (Uncertainty Shadow)
    # Visualizes the margin of error
    fig.add_trace(go.Scatter(
        x=pd.concat([pred_data['ds'], pred_data['ds'][::-1]]),
        y=pd.concat([pred_data['yhat_upper'], pred_data['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(255, 75, 75, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name="Confidence Interval"
    ))

    # D. Layout Styling (Dark Mode / Cyber Theme)
    fig.update_layout(
        title="Predictive Resource Demand (Next 30 Days)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0', family="sans-serif"),
        xaxis=dict(
            showgrid=False, 
            title=None,
            tickformat="%b %d"
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#333333', 
            title='Daily Updates Volume',
            zerolinecolor='#333333'
        ),
        legend=dict(
            orientation="h", 
            y=1.1,
            font=dict(size=12)
        ),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
        height=350
    )

    return fig

# Unit Test
if __name__ == "__main__":
    print("Testing Forecaster...")
    try:
        fig = generate_forecast_plot()
        if fig:
            print("✅ Forecast generated successfully.")
        else:
            print("❌ Forecast generation failed. Check 'data/daily_timeseries.csv'.")
    except Exception as e:
        print(f"❌ Error during test: {e}")