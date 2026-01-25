
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class Policy(ABC):
    """
    Abstract Base Class for all trading policies.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        """
        Initialize the policy.

        Args:
            cfg (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
        """
        self.cfg = cfg or {}

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """
        Takes an observation and returns an action.

        Args:
            observation (Any): The current state of the environment.

        Returns:
            Any: The action to take (e.g., Hold/Buy/Sell).
        """
        pass

    def __call__(self, observation: Any) -> Any:
        """
        Allows calling the policy object directly.
        """
        return self.act(observation)

    def reset(self) -> None:
        """
        Optional reset method for stateful policies.
        """
        pass


class ConstructivePolicy(nn.Module, ABC):
    """Base class for neural constructive policies."""

    @abstractmethod
    def forward(self, td: Any, env: Any, decode_type: str = "sampling", **kwargs: Any) -> Any:
        """Forward pass for policy."""


class ImprovementPolicy(nn.Module, ABC):
    """Base class for iterative improvement policies."""

    @abstractmethod
    def forward(self, td: Any, env: Any, **kwargs: Any) -> Any:
        """Iterative update pass."""
