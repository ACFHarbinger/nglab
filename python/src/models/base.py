from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["BaseModel", "BaseEncoder", "BaseDecoder", "BaseEmbedding"]


class BaseModel(nn.Module, ABC):
    """Base class for all models."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass logic."""
        pass


class BaseEncoder(BaseModel):
    """Base encoder interface."""

    @abstractmethod
    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Encode input."""
        pass


class BaseDecoder(BaseModel):
    """Base decoder interface."""

    @abstractmethod
    def forward(self, x: Tensor, context: Any = None, **kwargs: Any) -> Tensor:
        """Decode input with context."""
        pass


class BaseEmbedding(BaseModel):
    """Base embedding layer interface."""

    @abstractmethod
    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Embed input."""
        pass
