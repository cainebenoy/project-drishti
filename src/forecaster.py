import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import os

def generate_forecast(days=30):
    """
    Trains a Prophet model on daily update volume and predicts 'days' into the future.
    Returns: A Plotly Figure object.
    """
    data_path = "data/daily_timeseries.csv"
    if not os.path.exists(data_path):
        return None

    # Load Data
    df = pd.read_csv(data_path)
    
    # Sanity Check
    if len(df) < 5: # Need minimal points to train
        return None

    # Train Prophet
    m = Prophet(daily_seasonality=True)
    m.fit(df)

    # Forecast
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)

    # Visualization (Custom Dark Mode Plot)
    fig = go.Figure()

    # Historical Data
    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['y'],
        mode='lines', name='Historical Volume',
        line=dict(color='#00FFFF', width=2)
    ))

    # Prediction
    pred_data = forecast.tail(days)
    fig.add_trace(go.Scatter(
        x=pred_data['ds'], y=pred_data['yhat'],
        mode='lines', name='AI Prediction',
        line=dict(color='#FF4B4B', width=2, dash='dot')
    ))

    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=pd.concat([pred_data['ds'], pred_data['ds'][::-1]]),
        y=pd.concat([pred_data['yhat_upper'], pred_data['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(255, 75, 75, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    fig.update_layout(
        title="Predictive Resource Demand (Next 30 Days)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#333333', title='Total Daily Updates'),
        legend=dict(orientation="h", y=1.1)
    )

    return fig