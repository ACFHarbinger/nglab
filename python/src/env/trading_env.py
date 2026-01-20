"""
Trading Environment for NGLab.

Provides a Gymnasium-compatible interface for simulating trading scenarios,
serving as the primary interface between agents and the market simulator.
"""

from typing import Any, ClassVar, Tuple, Optional, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    A placeholder Trading Environment following Gymnasium API.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human"]}

    def __init__(self, lookback: int = 30, max_steps: int = 1000, feature_dim: int = 12) -> None:
        """
        Initialize the trading environment.

        Args:
            lookback (int): Number of historical steps in observation.
            max_steps (int): Maximum steps per episode.
            feature_dim (int): Number of features per step.
        """
        super().__init__()
        self.lookback = lookback
        self.max_steps = max_steps
        self.feature_dim = feature_dim

        # Action space: Buy, Sell, Hold (e.g. Discrete(3) or Continuous)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32
        )

        self.current_step = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment state.
        """
        super().reset(seed=seed)
        self.current_step = 0
        observation = self.observation_space.sample()  # Placeholder
        info: Dict[str, Any] = {}
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        """
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        reward = float(np.random.randn())  # Placeholder
        observation = self.observation_space.sample()
        info: Dict[str, Any] = {}
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """
        Render the environment.
        """
        pass

    def close(self) -> None:
        """
        Close the environment.
        """
        pass
