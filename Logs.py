import streamlit as st
from datetime import datetime

st.title("📜 Live Bot Logs")

log_container = st.container()

logs = [
    "Bot started successfully",
    "Strategy 'RSI + MACD' loaded",
    "BUY signal → 0.03 BTC @ 71,240",
    "Order filled @ 71,238",
    "Take profit hit → +2.8% ($84)",
    "Heartbeat OK • Equity: $52,847",
]

with log_container:
    for log in reversed(logs):
        st.code(f"[{datetime.now().strftime('%H:%M:%S')}] {log}", language=None)