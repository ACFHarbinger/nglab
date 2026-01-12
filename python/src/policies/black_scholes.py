import numpy as np
from scipy.stats import norm
from .base import Policy

class BlackScholesPolicy(Policy):
    """
    Policy based on Black-Scholes option pricing model.
    Decides to buy or sell based on the theoretical price vs market price.
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.risk_free_rate = self.cfg.get('risk_free_rate', 0.05)
        self.volatility = self.cfg.get('volatility', 0.2)
        
    def _black_scholes_call(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    def act(self, observation):
        # Observation is expected to be a dict or object with:
        # price, strike, time_to_maturity
        
        # Simplified parsing
        if isinstance(observation, dict):
             S = observation.get('price', 100)
             K = observation.get('strike', 100)
             T = observation.get('time_to_maturity', 1.0)
        else:
             # Assuming tensor or array: [Price, Strike, TTM]
             S, K, T = observation[0], observation[1], observation[2]

        theoretical_price = self._black_scholes_call(S, K, T, self.risk_free_rate, self.volatility)
        
        # Simple logic: Buy if undervalued, Sell if overvalued
        # Action: 0=Hold, 1=Buy, 2=Sell
        if S < theoretical_price * 0.95:
            return 1 # Buy
        elif S > theoretical_price * 1.05:
            return 2 # Sell
        else:
            return 0 # Hold
