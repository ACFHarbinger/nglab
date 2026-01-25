
from __future__ import annotations

from typing import Any

from python.src.utils.registry import ENV_REGISTRY


class EnvFactory:
    """Factory for creating environment instances."""

    @staticmethod
    def get_env(env_name: str, **kwargs: Any) -> Any:
        """Get environment by name from registry."""
        try:
            env_cls = ENV_REGISTRY.get(env_name.lower())
            return env_cls(**kwargs)
        except ValueError as e:
            # Re-raise with factory context if needed
            raise e
