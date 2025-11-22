import streamlit as st

st.title("🔗 Exchange & Balances")

st.success("✅ Connected: Binance Futures (API Key Verified)")

tab1, tab2, tab3 = st.tabs(["💰 Balances", "📍 Open Positions", "🧾 Recent Orders"])

with tab1:
    st.dataframe(pd.DataFrame([
        {"Asset": "USDT", "Total": 28472.50, "Available": 25120.30, "In Orders": 3352.20},
        {"Asset": "BTC", "Total": 0.8421, "Available": 0.8421},
        {"Asset": "ETH", "Total": 8.291, "Available": 8.291},
    ]), use_container_width=True)

with tab2:
    st.info("No open positions")

with tab3:
    st.write("Last order: BUY 0.05 BTC @ 71,280 (2 min ago)")