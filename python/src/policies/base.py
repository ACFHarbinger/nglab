from __future__ import annotations

import torch.nn as nn
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from tensordict import TensorDict
    from python.src.envs.base import TradingEnvBase

__all__ = ["Policy", "ConstructivePolicy", "ImprovementPolicy"]


class Policy(ABC):
    """
    Abstract Base Class for all trading policies.
    """

    name: str = "base"

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
    """
    Base class for constructive policies with multiple inheritance.
    Constructs a solution or action step-by-step.
    """

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        env: Optional[TradingEnvBase] = None,
        decode_type: str = "sampling",
        **kwargs: Any,
    ) -> Any:
        """Forward pass for policy."""
        pass

    @abstractmethod
    def evaluate(self, td: TensorDict, **kwargs: Any) -> Any:
        """Evaluate policy without exploration."""
        pass


class ImprovementPolicy(nn.Module, ABC):
    """
    Base class for policies that improve an existing solution.
    Used for local search or iterative refinement.
    """

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        env: Optional[TradingEnvBase] = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass for improvement policy."""
        pass
