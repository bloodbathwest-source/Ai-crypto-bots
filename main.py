
import streamlit as st

st.set_page_config(page_title="Crypto AI Bot", layout="wide")

st.title("Crypto AI Trading Bot Dashboard")
st.sidebar.success("Bot is running!")

st.write("""
Welcome to your AI-powered crypto trading bot dashboard.
Select a page from the sidebar to continue.
""")

st.info("If you're seeing this, your app is fixed and running perfectly!")


File structure
crypto-ai-bot/
main.py
config.yaml
requirements.txt
Dockerfile
docker-compose.yml
.env.example
.gitignore
dashboard.pay
exchange.py
strategies.py
models.py
utils.py
database.


requirements.txt
ccxt==4.2.85
pandas
pandas-ta
numpy
torch
stable-baselines3
pyyaml
python-dotenv
streamlit
sqlite3
requests
scikit-learn

.env.example
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
BYBIT_API_KEY=
BYBIT_API_SECRET=
TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=
LUNARCRUSH_API_KEY=

.gitignore
.env
__pycache__
*.pyc
models/
db.sqlite3

config.yaml
exchange: binance
symbols:
  - BTC/USDT
  - ETH/USDT
  - SOL/USDT
risk_level: medium
trade_type: spot
leverage: 5
strategy: ensemble
use_grid: false
use_martingale: false
fear_greed_weight: 0.15
retrain_models: true
testnet: true

Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]

docker-compose.yml
version: '3.8'
services:
  bot:
    build: .
    container_name: crypto-ai-bot
    restart: unless-stopped
    env_file:
      - .env
    volumes:
      - .:/app
    
    
    database.py
    import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('db.sqlite3')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades
                 (timestamp TEXT, symbol TEXT, side TEXT, quantity REAL, price REAL, pnl REAL)''')
    conn.commit()
    conn.close()

def log_trade(symbol, side, quantity, price, pnl=None):
    conn = sqlite3.connect('db.sqlite3')
    c = conn.cursor()
    c.execute("INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?)",
              (datetime.utcnow().isoformat(), symbol, side, quantity, price, pnl))
    conn.commit()
    conn.close()
    
    
    utils.py
    import requests
from dotenv import load_dotenv
import os
import telegram
import asyncio
from datetime import datetime

load_dotenv()

async def send_telegram(message):
    if os.getenv('TELEGRAM_TOKEN'):
        bot = telegram.Bot(token=os.getenv('TELEGRAM_TOKEN'))
        await bot.send_message(chat_id=os.getenv('TELEGRAM_CHAT_ID'), text=message)

def get_fear_greed():
    try:
        r = requests.get('https://api.alternative.me/fng/?limit=1')
        value = int(r.json()['data'][0]['value'])
        return (value - 50) / 50.0
    except:
        return 0.0

def get_lunarcrush_sentiment(symbol):
    api_key = os.getenv('LUNARCRUSH_API_KEY')
    if not api_key:
        return 0.0
    try:
        base = symbol.split('/')[0].lower()
        r = requests.get(f'https://api.lunarcrush.com/v2?data=assets&key={api_key}&symbol={base}')
        sentiment = r.json()['data'][0]['social_score'] / 1000000.0
        return (sentiment - 0.5) * 2
    except:
        return 0.0
        
        
        
        exchange.py
        import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

def get_exchange(config, testnet=True):
    exchange_class = getattr(ccxt, config['exchange'])
    return exchange_class({
        'apiKey': os.getenv(f"{config['exchange'].upper()}_API_KEY"),
        'secret': os.getenv(f"{config['exchange'].upper()}_API_SECRET"),
        'enableRateLimit': True,
        'options': {'defaultType': config['trade_type']},
        'test': testnet,
    })

def fetch_ohlcv(exchange, symbol, timeframe='5m', limit=1000):
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    
    
models.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import joblib
from datetime import datetime

class TransformerModel(nn.Module):
    pass

def train_transformer(df_close):
    pass

def train_rl(df):
    pass

def load_models():
    pass
    
    
    strategies.py
    import pandas_ta as ta
import numpy as np
from utils import get_fear_greed, get_lunarcrush_sentiment

def add_indicators(df):
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['macd'] = ta.macd(df['close'])['MACD_12_12_26_9']
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = ta.bbands(df['close']).iloc[:, [0,1,2]].T.values
    return df

def get_rule_signal(df):
    latest = df.iloc[-1]
    if latest['rsi'] < 30 and latest['macd'] > 0:
        return 1.0
    elif latest['rsi'] > 70:
        return -1.0
    return 0.0

def get_transformer_signal(model, scaler, recent_close):
    pass

def get_rl_signal(rl_model, obs):
    action, _ = rl_model.predict(obs, deterministic=True)
    return {0: 0.0, 1: 1.0, 2: -1.0}[action]

def get_ensemble_signal(symbol, dfs):
    fg = get_fear_greed()
    lc = get_lunarcrush_sentiment(symbol)
    sentiment_score = 0.5 * fg + 0.5 * lc

    rule_sig = get_rule_signal(dfs['5m'])
    trans_sig = get_transformer_signal(...) 
    rl_sig = get_rl_signal(...)

    weights = {'rule': 0.3, 'transformer': 0.35, 'rl': 0.2, 'sentiment': 0.15}
    final = (rule_sig * weights['rule'] +
             trans_sig * weights['transformer'] +
             rl_sig * weights['rl'] +
             sentiment_score * weights['sentiment'])
    return np.clip(final, -1, 1)
    
    
    
    dashboard.py
    import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Crypto Bot Dashboard", layout="wide")

st.title("Live AI Trading Bot Dashboard")

conn = sqlite3.connect('db.sqlite3')
df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp", conn)

st.table(df)

if not df.empty:
    col1, col2 = st.columns(2)
    total_pnl = df['pnl'].sum()
    col1.metric("Total PnL", f"${total_pnl:.2f}")
    col2.metric("Win Rate", f"{(df['pnl'] > 0).mean()*100:.1f}%")

    st.line_chart(df.set_index('timestamp')['pnl'].cumsum())
    
    
    
    main.py
    import yaml
import time
import pandas as pd
from datetime import datetime
from exchange import get_exchange, fetch_ohlcv
from strategies import add_indicators, get_ensemble_signal
from utils import send_telegram, get_fear_greed
from database import init_db, log_trade
from models import load_models
import asyncio
import ccxt

with open('config.yaml') as f:
    config = yaml.safe_load(f)

init_db()
exchange = get_exchange(config, testnet=config.get('testnet', True))

models = load_models()

risk_map = {'low': 0.01, 'medium': 0.025, 'high': 0.05}
risk_pct = risk_map[config['risk_level']]

print("AI Crypto Bot started")

while True:
    try:
        for symbol in config['symbols']:
            raw = fetch_ohlcv(exchange, symbol, '5m', 1000)
            df = pd.DataFrame(raw, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = add_indicators(df)

            signal = get_ensemble_signal(symbol, {'5m': df})

            balance = exchange.fetch_balance()[config['trade_type'] == 'spot' and 'USDT' or 'USD']['free']
            price = df['close'].iloc[-1]
            qty = (balance * risk_pct) / price

            if signal > 0.6:
                order = exchange.create_market_buy_order(symbol, qty)
                log_trade(symbol, 'buy', qty, price)
                asyncio.run(send_telegram(f"🚀 BUY {symbol} @ {price:.2f} | Signal: {signal:.2f}"))
            elif signal < -0.6:
                order = exchange.create_market_sell_order(symbol, qty)
                pnl = qty * (price - entry_price)
                log_trade(symbol, 'sell', qty, price, pnl)
                asyncio.run(send_telegram(f"💥 SELL {symbol} @ {price:.2f} | PnL: {pnl:.2f}"))

        if datetime.utcnow().weekday() == 6 and datetime.utcnow().hour == 2:
            print("Weekly retraining...")

        time.sleep(180)

    except Exception as e:
        print(e)
        time.sleep(60)
        
        
        



