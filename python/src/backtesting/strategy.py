"""
Strategy interface and base class for backtesting.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Strategy(Protocol):
    """Protocol defining the interface for a trading strategy."""

    def on_market_data(self, market_id: str, price: float, timestamp: int) -> None:
        """React to a market price update."""
        ...

    def on_fill(self, market_id: str, amount: float, price: float, side: str) -> None:
        """React to an order fill."""
        ...


class BaseStrategy(ABC):
    """Base class for implementing trading strategies."""

    def __init__(self, name: str = "BaseStrategy"):
        """Initialize BaseStrategy."""
        self.name = name
        self.engine: Any | None = None

    def set_engine(self, engine: Any) -> None:
        """Link the strategy to the backtest engine."""
        self.engine = engine

    @abstractmethod
    def on_market_data(self, market_id: str, price: float, timestamp: int) -> None:
        """Implement strategy logic here."""
        pass

    def on_fill(  # noqa: B027
        self, market_id: str, amount: float, price: float, side: str
    ) -> None:
        """Optional callback for order fills."""
        pass
