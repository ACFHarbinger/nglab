"""
Threshold-based Policy for NGLab.

Simple rule-based strategy that buys/sells based on fixed price thresholds.
"""

from .base import Policy


class ThresholdPolicy(Policy):
    """
    Simple threshold-based policy.
    Buy if price < buy_threshold.
    Sell if price > sell_threshold.
    """

    def __init__(self, cfg=None):
        """
        Initialize Threshold policy.

        Args:
            cfg (Dict, optional): Configuration containing buy/sell thresholds.
        """
        super().__init__(cfg)
        self.buy_threshold = self.cfg.get("buy_threshold", 90.0)
        self.sell_threshold = self.cfg.get("sell_threshold", 110.0)

    def act(self, observation):
        """
        Determine action based on price relative to thresholds.
        """
        # Observation assumed to contain 'price' or be a price value
        if isinstance(observation, dict):
            price = observation.get("price", 100)
        else:
            price = (
                observation if isinstance(observation, (int, float)) else observation[0]
            )

        if price < self.buy_threshold:
            return 1  # Buy
        elif price > self.sell_threshold:
            return 2  # Sell
        return 0  # Hold
