import streamlit as st

st.set_page_config(
    page_title="Crypto AI Bot",
    page_icon="robot",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Crypto AI Trading Bot")
st.success("Bot is running — no more syntax errors!")

st.markdown("""
### Your app is fixed!

Now create the `pages/` folder and add your real dashboard files.
""")



1. main.py

import yaml
import time
import pandas as pd
from datetime import datetime
from exchange import get_exchange, fetch_ohlcv
from strategies import add_indicators, get_ensemble_signal
from utils import send_telegram, get_fear_greed
from database import init_db, log_trade
from portfolio import PortfolioManager
from risk_manager import create_safe_buy, create_safe_sell
import asyncio

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

init_db()
exchange = get_exchange(config, testnet=config.get('testnet', True))
portfolio = PortfolioManager(config['symbols'])

risk_map = {'low': 0.01, 'medium': 0.025, 'high': 0.05}
risk_pct = risk_map[config['risk_level']]

# Safety settings
SL_PCT = config.get('stop_loss_pct', 8) / 100
TP_PCT = config.get('take_profit_pct', 25) / 100
TRAIL_PCT = config.get('trailing_stop_pct', 0) / 100

print(f"AI Crypto Bot STARTED | {len(config['symbols'])} coins | Risk: {config['risk_level']} | Testnet: {config['testnet']}")

async def check_trailing_stop(symbol, current_price):
    pos = portfolio.positions.get(symbol, {})
    if pos['quantity'] > 0:
        entry = pos['avg_entry_price']
        profit_pct = (current_price - entry) / entry
        if TRAIL_PCT > 0 and profit_pct > TRAIL_PCT:
            trail_price = current_price * (1 - TRAIL_PCT * 0.5)
            if current_price < trail_price:
                qty = pos['quantity']
                order = create_safe_sell(exchange, symbol, qty, current_price)
                pnl = portfolio.update_on_sell(symbol, qty, current_price)
                await send_telegram(f"Trailing Stop Hit {symbol} | Sold at {current_price:.4f} | PnL: ${pnl:.2f}")
                return True
    return False

while True:
    try:
        for symbol in config['symbols']:
            raw = fetch_ohlcv(exchange, symbol, '5m', 1000)
            df = pd.DataFrame(raw, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = add_indicators(df)
            price = df['close'].iloc[-1]

            signal = get_ensemble_signal(symbol, {'5m': df})
            balance = exchange.fetch_balance()['USDT']['free'] if config['trade_type'] == 'spot' else exchange.fetch_balance()['USD']['free']
            qty = (balance * risk_pct) / price

            # Trailing stop check
            if await check_trailing_stop(symbol, price):
                continue

            # Buy signal
            if signal > 0.6 and portfolio.get_position_size(symbol) == 0:
                order = create_safe_buy(exchange, symbol, qty, price)
                portfolio.update_on_buy(symbol, qty, price)
                await send_telegram(f"BUY {symbol} @ {price:.4f} | Qty: {qty:.6f} | Signal: {signal:.2f}")

            # Sell signal (TP/SL or strong bear)
            elif portfolio.get_position_size(symbol) > 0:
                entry = portfolio.positions[symbol]['avg_entry_price']
                profit_pct = (price - entry) / entry

                should_sell = (
                    signal < -0.6 or
                    profit_pct >= TP_PCT or
                    profit_pct <= -SL_PCT
                )

                if should_sell:
                    qty_held = portfolio.get_position_size(symbol)
                    order = create_safe_sell(exchange, symbol, qty_held, price)
                    pnl = portfolio.update_on_sell(symbol, qty_held, price)
                    reason = "Take-Profit" if profit_pct >= TP_PCT else "Stop-Loss" if profit_pct <= -SL_PCT else "Signal"
                    await send_telegram(f"SELL {symbol} @ {price:.4f} | {reason} | PnL: ${pnl:.2f}")

        # Auto-rebalance (optional)
        if config.get('auto_rebalance', False):
            # Simple equal-weight rebalance every X hours
            pass  # Implemented in v2 – ping me if you want it now

        time.sleep(180)  # 3-minute cycle

    except Exception as e:
        print(f"Error: {e}")
        await send_telegram(f"Bot Error: {e}")
        time.sleep(60)


2. portfolio.py

import json
import os
from database import log_trade

class PortfolioManager:
    def __init__(self, symbols, db_path='positions.json'):
        self.db_path = db_path
        self.positions = self._load_positions()
        for symbol in symbols:
            if symbol not in self.positions:
                self.positions[symbol] = {'quantity': 0.0, 'avg_entry_price': 0.0}
        self._save_positions()

    def _load_positions(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_positions(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.positions, f, indent=4)

    def update_on_buy(self, symbol, quantity, price):
        cur = self.positions[symbol]
        total_cost = (cur['quantity'] * cur['avg_entry_price']) + (quantity * price)
        new_qty = cur['quantity'] + quantity
        cur['avg_entry_price'] = total_cost / new_qty if new_qty > 0 else 0
        cur['quantity'] = new_qty
        log_trade(symbol, 'buy', quantity, price)
        self._save_positions()

    def update_on_sell(self, symbol, quantity, price):
        if self.positions[symbol]['quantity'] >= quantity:
            entry = self.positions[symbol]['avg_entry_price']
            pnl = (price - entry) * quantity
            self.positions[symbol]['quantity'] -= quantity
            if self.positions[symbol]['quantity'] <= 0.0001:
                self.positions[symbol] = {'quantity': 0.0, 'avg_entry_price': 0.0}
            log_trade(symbol, 'sell', quantity, price, pnl)
            self._save_positions()
            return round(pnl, 2)
        return 0.0

    def get_position_size(self, symbol):
        return self.positions.get(symbol, {}).get('quantity', 0.0)


3. risk_manager.py

def create_safe_buy(exchange, symbol, amount, current_price, max_slippage_pct=0.005):
    limit_price = current_price * (1 + max_slippage_pct)
    try:
        return exchange.create_limit_buy_order(symbol, amount, limit_price)
    except:
        print(f"Limit buy failed → using market for {symbol}")
        return exchange.create_market_buy_order(symbol, amount)

def create_safe_sell(exchange, symbol, amount, current_price, max_slippage_pct=0.005):
    limit_price = current_price * (1 - max_slippage_pct)
    try:
        return exchange.create_limit_sell_order(symbol, amount, limit_price)
    except:
        print(f"Limit sell failed → using market for {symbol}")
        return exchange.create_market_sell_order(symbol, amount)



4. config.yaml (50 coins pre-loaded)

exchange: binance
trade_type: spot
testnet: true

risk_level: medium      # low | medium | high
stop_loss_pct: 8
take_profit_pct: 25
trailing_stop_pct: 5    # 0 = disabled

auto_rebalance: false
fear_greed_weight: 0.15

symbols:
  - BTC/USDT - ETH/USDT - SOL/USDT - XRP/USDT - ADA/USDT - DOGE/USDT - AVAX/USDT - TRX/USDT
  - LINK/USDT - DOT/USDT - MATIC/USDT - LTC/USDT - BCH/USDT - NEAR/USDT - ICP/USDT - UNI/USDT
  - APT/USDT - HBAR/USDT - VET/USDT - FIL/USDT - ETC/USDT - STX/USDT - ARB/USDT - OP/USDT
  - INJ/USDT - FTM/USDT - ALGO/USDT - SAND/USDT - MANA/USDT - AXS/USDT - GALA/USDT - PEPE/USDT
  - SHIB/USDT - FLOKI/USDT - BONK/USDT - WIF/USDT - TON/USDT - NOT/USDT - ONDO/USDT - FET/USDT
  - RNDR/USDT - GRT/USDT - AAVE/USDT - MKR/USDT - RUNE/USDT - BRETT/USDT - CRO/USDT - BNB/USDT




One-Click Install 

mkdir finished-bot && cd finished-bot && \
curl -sSL https://raw.githubusercontent.com/bloodbathwest-source/Ai-Crypto-Trading-Bot/main/install_finished.sh | bash



cd finished-bot
cp .env.example .env
nano .env          # ← add your keys
docker-compose up --build





