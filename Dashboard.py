import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

st.title("📊 Live Trading Dashboard")

# Mock real-time data
@st.cache_data(ttl=5)  # Updates every 5 seconds
def get_live_data():
    now = datetime.now()
    equity = 50000 + sum(random.gauss(0, 150) for _ in range(1000))
    return {
        "equity": equity,
        "daily_pnl": random.uniform(-800, 2200),
        "win_rate": round(random.uniform(58, 78), 1),
        "trades_today": random.randint(8, 47),
        "uptime": (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)).seconds // 60
    }

data = get_live_data()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Equity", f"${data['equity']:,.2f}", f"{data['daily_pnl']:+,.0f}")
with col2:
    st.metric("Win Rate", f"{data['win_rate']}%", "↑ 2.4%")
with col3:
    st.metric("Trades Today", data['trades_today'])
with col4:
    st.metric("Bot Uptime", f"{data['uptime']} min")

# Equity curve
dates = pd.date_range(end=datetime.now(), periods=200, freq='30min')
equity = 50000 + pd.np.cumsum(pd.np.random.randn(200) * 180)
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=equity, mode='lines', name='Equity', line=dict(color='#00D4A4')))
fig.update_layout(title="Equity Curve (Live)", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)