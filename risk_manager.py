# risk_manager.py
def create_safe_buy(exchange, symbol, amount, current_price, max_slippage_pct=0.005):
    limit_price = current_price * (1 + max_slippage_pct)
    try:
        return exchange.create_limit_buy_order(symbol, amount, limit_price)
    except Exception as e:
        print(f"Limit buy failed ({e}), using market order")
        return exchange.create_market_buy_order(symbol, amount)

def create_safe_sell(exchange, symbol, amount, current_price, max_slippage_pct=0.005):
    limit_price = current_price * (1 - max_slippage_pct)
    try:
        return exchange.create_limit_sell_order(symbol, amount, limit_price)
    except Exception as e:
        print(f"Limit sell failed ({e}), using market order")
        return exchange.create_market_sell_order(symbol, amount)