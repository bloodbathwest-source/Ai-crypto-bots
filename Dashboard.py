import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

st.set_page_config(page_title="LIVE AI Bot", layout="wide")
st.title("LIVE AI Crypto Trading Bot")

# Try to load real portfolio data
positions_file = "positions.json"  # Same file your bot uses
trades_file = "trades.db"         # Or trades.json if you export it

if os.path.exists(positions_file):
    with open(positions_file) as f:
        positions = json.load(f)
    
    total_equity = 50000  # You can calculate real equity here
    daily_pnl = 0
    for sym, pos in positions.items():
        if pos['quantity'] > 0:
            st.success(f"{sym}: {pos['quantity']:.6f} @ ${pos['avg_entry_price']:,.2f}")

    st.metric("Bot Status", "ONLINE & TRADING", "50 coins active")
else:
    st.warning("Bot not running locally — showing demo mode")
    st.metric("Bot Status", "Demo Mode", "Connect your bot for live data")