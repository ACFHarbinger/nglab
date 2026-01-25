"""
Threshold-based Policy for NGLab.

Simple rule-based strategy that buys/sells based on fixed price thresholds.
"""

from typing import Any

from python.src.utils.registry import register_policy
from .base import Policy


@register_policy("threshold")
class ThresholdPolicy(Policy):
    """
    Simple threshold-based policy.
    Buy if price < buy_threshold.
    Sell if price > sell_threshold.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        """
        Initialize Threshold policy.

        Args:
            cfg (Dict, optional): Configuration containing buy/sell thresholds.
        """
        super().__init__(cfg)
        self.buy_threshold = float(self.cfg.get("buy_threshold", 90.0))
        self.sell_threshold = float(self.cfg.get("sell_threshold", 110.0))

    def act(self, observation: Any) -> int:
        """
        Determine action based on price relative to thresholds.
        """
        # Observation assumed to contain 'price' or be a price value
        if isinstance(observation, dict):
            price = float(observation.get("price", 100.0))
        elif isinstance(observation, (int, float)):
            price = float(observation)
        elif hasattr(observation, "__getitem__"):
            price = float(observation[0])
        else:
            price = 100.0

        if price < self.buy_threshold:
            return 1  # Buy
        elif price > self.sell_threshold:
            return 2  # Sell
        return 0  # Hold
