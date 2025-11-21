import json
import os
from database import log_trade

class PortfolioManager:
    def __init__(self, symbols, db_path='positions.json'):
        self.db_path = db_path
        self.positions = self._load_positions()
        for symbol in symbols:
            if symbol not in self.positions:
                self.positions[symbol] = {'quantity': 0, 'avg_entry_price': 0}
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
        current_qty = self.positions[symbol]['quantity']
        current_avg_price = self.positions[symbol]['avg_entry_price']

        total_cost = (current_qty * current_avg_price) + (quantity * price)
        new_qty = current_qty + quantity
        self.positions[symbol]['avg_entry_price'] = total_cost / new_qty
        self.positions[symbol]['quantity'] = new_qty
        
        log_trade(symbol, 'buy', quantity, price)
        self._save_positions()
        print(f"Updated position for {symbol}: {self.positions[symbol]}")

    def update_on_sell(self, symbol, quantity, price):
        if self.positions[symbol]['quantity'] > 0:
            entry_price = self.positions[symbol]['avg_entry_price']
            pnl = (price - entry_price) * quantity
            
            self.positions[symbol]['quantity'] -= quantity
            if self.positions[symbol]['quantity'] <= 0:
                 self.positions[symbol]['avg_entry_price'] = 0

            log_trade(symbol, 'sell', quantity, price, pnl)
            self._save_positions()
            print(f"Closed position for {symbol}. PnL: {pnl:.2f}")
            return pnl
        return 0.0

    def get_position_size(self, symbol):
        return self.positions.get(symbol, {}).get('quantity', 0)