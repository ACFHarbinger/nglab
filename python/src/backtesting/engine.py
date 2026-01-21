"""
Core backtesting engine wrapping PolymarketArena.
"""

import json
from typing import Any

from nglab import PolymarketArena

from .strategy import Strategy


class BacktestEngine:
    """
    Python wrapper for driving backtests using PolymarketArena.
    """

    def __init__(self, initial_collateral: float = 10000.0, taker_fee: float = 0.001):
        """Initialize BacktestEngine."""
        self.arena = PolymarketArena(initial_collateral, taker_fee)
        self.strategy: Strategy | None = None
        self.market_ids: list[str] = []
        self.history: list[dict[str, Any]] = []

    def set_strategy(self, strategy: Strategy) -> None:
        """Assign a strategy to the engine."""
        self.strategy = strategy
        if hasattr(strategy, "set_engine"):
            strategy.set_engine(self)

    def load_data(self, markets_json: str, price_histories: dict[str, str]) -> None:
        """
        Load market metadata and price history into the arena.

        Args:
            markets_json: JSON string containing market metadata.
            price_histories: Dict mapping market_id to CSV string data.
        """
        self.arena.load_markets(markets_json)

        # Extract market IDs from JSON to facilitate looping
        try:
            markets = json.loads(markets_json)
            self.market_ids = [str(m["id"]) for m in markets]
        except Exception as e:
            print(f"Warning: Failed to parse markets_json for tracking: {e}")

        for market_id, csv_data in price_histories.items():
            self.arena.load_price_history(market_id, csv_data)

    def run(self) -> list[dict[str, Any]]:
        """
        Execute the backtest loop.
        """
        if not self.strategy:
            raise ValueError("No strategy assigned to the engine.")

        self.history = []

        running = True
        while running:
            # 1. Update strategy with current prices
            for market_id in self.market_ids:
                price = self.arena.get_price(market_id)
                if price is not None:
                    # In a real event-driven engine, we'd only signal changes,
                    # but for replay we can signal every tick.
                    self.strategy.on_market_data(
                        market_id, price, self.arena.current_step()
                    )

            # 2. Record state
            self.history.append(
                {
                    "step": self.arena.current_step(),
                    "collateral": self.arena.collateral(),
                    "account_value": self.arena.account_value(),
                    "realized_pnl": self.arena.realized_pnl(),
                }
            )

            # 3. Advance to next tick
            running = self.arena.advance()

        return self.history

    # --- Proxy methods to Arena ---

    def buy_yes(self, market_id: str, amount: float) -> float:
        """Buy Yes tokens."""
        cost = self.arena.buy_yes(market_id, amount)
        if self.strategy:
            price = self.arena.get_price(market_id)
            if price:
                self.strategy.on_fill(market_id, amount, price, "buy_yes")
        return cost

    def buy_no(self, market_id: str, amount: float) -> float:
        """Buy No tokens."""
        cost = self.arena.buy_no(market_id, amount)
        if self.strategy:
            price = self.arena.get_price(market_id)
            if price:
                self.strategy.on_fill(market_id, amount, 1.0 - price, "buy_no")
        return cost

    def sell_yes(self, market_id: str, amount: float) -> float:
        """Sell Yes tokens."""
        proceeds = self.arena.sell_yes(market_id, amount)
        if self.strategy:
            price = self.arena.get_price(market_id)
            if price:
                self.strategy.on_fill(market_id, amount, price, "sell_yes")
        return proceeds

    def split(self, market_id: str, amount: float) -> float:
        """Split collateral into complete sets."""
        return self.arena.split(market_id, amount)

    def merge(self, market_id: str, amount: float) -> float:
        """Merge complete sets back to collateral."""
        return self.arena.merge(market_id, amount)

    def get_position(self, market_id: str) -> tuple[float, float]:
        """Get (yes, no) tokens."""
        return self.arena.get_position(market_id)

    def get_price(self, market_id: str) -> float | None:
        """Get Yes price."""
        return self.arena.get_price(market_id)
