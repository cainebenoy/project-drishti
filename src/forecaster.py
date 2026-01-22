import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import os

def generate_forecast(days=30):
    """
    Trains a Prophet model on daily update volume and predicts 'days' into the future.
    Returns: A Plotly Figure object.
    """
    # 1. Path Verification
    data_path = "data/daily_timeseries.csv"
    if not os.path.exists(data_path):
        return None

    # 2. Data Loading & Cleaning
    try:
        df = pd.read_csv(data_path)
        # Ensure 'ds' is datetime
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        # Drop bad dates
        df = df.dropna(subset=['ds', 'y'])
        
        # Sanity Check: Need minimal points to train
        if len(df) < 5: 
            return None
    except Exception as e:
        print(f"Forecasting Data Error: {e}")
        return None

    # 3. Model Training
    # Prophet requires 'ds' (date) and 'y' (target value)
    try:
        m = Prophet(
            daily_seasonality=False,
            yearly_seasonality=False,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        m.fit(df)

        # 4. Future Dataframe Generation
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)
    except Exception as e:
        print(f"Prophet Training Error: {e}")
        return None

    # 5. Advanced Visualization (Cyber/Dark Mode)
    fig = go.Figure()

    # A. Historical Data (Cyan Line)
    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['y'],
        mode='lines', name='Historical Volume',
        line=dict(color='#00FFFF', width=2),
        opacity=0.8
    ))

    # B. AI Prediction (Red Dotted Line)
    pred_data = forecast.tail(days)
    fig.add_trace(go.Scatter(
        x=pred_data['ds'], y=pred_data['yhat'],
        mode='lines', name='AI Prediction',
        line=dict(color='#FF4B4B', width=3, dash='dot')
    ))

    # C. Confidence Interval (Shadow)
    # This visualizes the AI's uncertainty
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

    # D. Layout Styling
    fig.update_layout(
        title="🔮 Predictive Resource Demand (Next 30 Days)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0', family="sans-serif"),
        xaxis=dict(
            showgrid=False, 
            title="Date",
            tickformat="%b %d"
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#333333', 
            title='Total Daily Updates',
            zerolinecolor='#333333'
        ),
        legend=dict(
            orientation="h", 
            y=1.1,
            font=dict(size=12)
        ),
        hovermode="x unified"
    )

    return fig

# Test block
if __name__ == "__main__":
    fig = generate_forecast()
    if fig:
        print("✅ Forecast generated successfully.")
    else:
        print("❌ Forecast generation failed.")