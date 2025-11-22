import streamlit as st

st.title("🎯 Active Strategies")

strategies = {
    "RSI + MACD Reversal": {"status": "🟢 Running", "pnl": "+$2,847", "trades": 42},
    "AI LSTM Predictor": {"status": "🟡 Learning", "pnl": "+$1,204", "trades": 18},
    "Grid Bot (BTC/USDT)": {"status": "🟢 Running", "pnl": "+$892", "trades": 156},
    "Volume Breakout": {"status": "⚪ Paused", "pnl": "-$124", "trades": 9},
}

for name, info in strategies.items():
    with st.expander(f"{info['status']} {name} • PnL: {info['pnl']} ({info['trades']} trades)"):
        col1, col2 = st.columns(2)
        with col1:
            st.button(f"Restart", key=name)
            st.button(f"Pause", key=name+"p")
        with col2:
            st.button(f"View Backtest", key=name+"b")
            st.button(f"Edit Params", key=name+"e")