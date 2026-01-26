
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from tensordict import TensorDict

__all__ = ["TradingEnvBase", "EnvironmentProtocol", "SimulationProtocol"]


@runtime_checkable
class EnvironmentProtocol(Protocol):
    """Protocol for environment-like objects to ensure duck typing compatibility."""

    name: str

    @property
    def batch_size(self) -> torch.Size: ...

    def reset(self, seed: Optional[int] = None) -> TensorDict: ...

    def step(self, action: TensorDict) -> TensorDict: ...

    def get_reward(self, td: TensorDict) -> torch.Tensor: ...


@runtime_checkable
class SimulationProtocol(Protocol):
    """Protocol specifically for simulation engines."""

    def run_simulation(self, steps: int) -> TensorDict: ...


class TradingEnvBase(ABC):
    """Unified trading environment interface."""

    name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = kwargs
        self._batch_size: torch.Size = torch.Size([])

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: torch.Size | int | list[int] | tuple[int, ...]) -> None:
        """Set batch size with validation and error handling."""
        if not isinstance(value, torch.Size):
            if isinstance(value, int):
                value = torch.Size([value])
            elif isinstance(value, (list, tuple)):
                value = torch.Size(value)
            else:
                raise TypeError(
                    f"batch_size must be torch.Size, int, list, or tuple. "
                    f"Got: {type(value).__name__}"
                )

        if any(v <= 0 for v in value):
            raise ValueError(
                f"batch_size must contain positive values. Got: {value}"
            )

        self._batch_size = value

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> TensorDict:
        """Reset environment and return initial state."""
        pass

    @abstractmethod
    def step(self, action: TensorDict) -> TensorDict:
        """Execute action and return next state."""
        pass

    @abstractmethod
    def get_reward(self, td: TensorDict) -> torch.Tensor:
        """Calculate reward from state."""
        pass

    def render(self) -> None:
        """Optional render method."""
        pass
