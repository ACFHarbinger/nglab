"""
Simple SMA Crossover strategy for backtest verification.
"""

from python.src.backtesting.strategy import BaseStrategy


class SMACrossoverStrategy(BaseStrategy):
    """
    Very simple strategy that buys when price increases.
    """

    def __init__(self, amount_to_buy: float = 10.0):
        """Initialize SMACrossoverStrategy."""
        super().__init__("SMACrossover")
        self.amount_to_buy = amount_to_buy
        self.last_price: float | None = None

    def on_market_data(self, market_id: str, price: float, timestamp: int) -> None:
        """Handle incoming market data and execute buys/sells based on price trends."""
        if self.engine is None:
            return
        if self.last_price is not None:
            if price > self.last_price:
                # Price went up, buy Yes
                try:
                    self.engine.buy_yes(market_id, self.amount_to_buy)
                except Exception as e:
                    print(f"Failed to buy: {e}")
            elif price < self.last_price:
                # Price went down, sell Yes if we have it
                yes, _ = self.engine.get_position(market_id)
                if yes >= self.amount_to_buy:
                    self.engine.sell_yes(market_id, self.amount_to_buy)

        self.last_price = price

    def on_fill(self, market_id: str, amount: float, price: float, side: str) -> None:
        """Handle order fill notifications."""
        print(f"Fill: {side} {amount} @ {price} on {market_id}")
