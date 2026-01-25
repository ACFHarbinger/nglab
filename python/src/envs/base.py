
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from tensordict import TensorDict


class TradingEnvBase(ABC):
    """Unified trading environment interface."""

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = kwargs
        self._batch_size: torch.Size = torch.Size([])

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: torch.Size) -> None:
        if not isinstance(value, torch.Size):
            value = torch.Size(value if isinstance(value, (list, tuple)) else [value])
        self._batch_size = value

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> TensorDict:
        """Reset environment and return initial state."""

    @abstractmethod
    def step(self, action: TensorDict) -> TensorDict:
        """Execute action and return next state."""

    def render(self) -> None:
        """Optional render method."""
        pass
