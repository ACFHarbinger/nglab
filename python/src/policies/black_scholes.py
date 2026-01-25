"""
Black-Scholes Policy for NGLab.

Implements a trading strategy based on the Black-Scholes option pricing model,
buying or selling based on deviations from the theoretical fair value.
"""

from typing import Any

import numpy as np
import torch
from scipy.stats import norm

from python.src.utils.registry import register_policy
from .base import Policy


@register_policy("black_scholes")
class BlackScholesPolicy(Policy):
    """
    Policy based on Black-Scholes option pricing model.
    Decides to buy or sell based on the theoretical price vs market price.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        """
        Initialize Black-Scholes policy.

        Args:
            cfg (Dict, optional): Configuration containing risk-free rate and volatility.
        """
        super().__init__(cfg)
        self.risk_free_rate: float = self.cfg.get("risk_free_rate", 0.05)
        self.volatility: float = self.cfg.get("volatility", 0.2)

    def _black_scholes_call(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """
        Calculate the Black-Scholes call price.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return call_price

    def act(self, observation: dict[str, float] | Any) -> int:
        """
        Determine action based on Black-Scholes fair value.
        """
        # Observation is expected to be a dict or object with:
        # price, strike, time_to_maturity

        # Simplified parsing
        if isinstance(observation, dict):
            S = float(observation.get("price", 100.0))
            K = float(observation.get("strike", 100.0))
            T = float(observation.get("time_to_maturity", 1.0))
        elif isinstance(observation, np.ndarray | torch.Tensor):
            # Assuming tensor or array: [Price, Strike, TTM]
            S, K, T = (
                float(observation[0]),
                float(observation[1]),
                float(observation[2]),
            )
        else:
            # Fallback for other sequence types
            S, K, T = (
                float(observation[0]),
                float(observation[1]),
                float(observation[2]),
            )

        theoretical_price = self._black_scholes_call(
            S, K, T, self.risk_free_rate, self.volatility
        )

        # Simple logic: Buy if undervalued, Sell if overvalued
        # Action: 0=Hold, 1=Buy, 2=Sell
        if S < theoretical_price * 0.95:
            return 1  # Buy
        elif S > theoretical_price * 1.05:
            return 2  # Sell
        else:
            return 0  # Hold
