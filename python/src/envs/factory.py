
from __future__ import annotations

from typing import Any

from python.src.envs.env_wrapper import TradingEnvWrapper
from python.src.envs.trading_env import TradingEnv


class EnvFactory:
    """Factory for creating environment instances."""

    @staticmethod
    def get_env(env_name: str, **kwargs: Any) -> Any:
        """Get environment by name."""
        name = env_name.lower()
        if "trading" in name:
            return TradingEnv(**kwargs)
        elif "wrapper" in name:
            return TradingEnvWrapper(**kwargs)
        else:
            raise ValueError(
                f"Unknown environment: {env_name}. "
                f"Available: trading, wrapper"
            )
