from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch.nn as nn

if TYPE_CHECKING:
    import torch
    from tensordict import TensorDict

__all__ = ["ModelProtocol", "BaseModel", "BaseEncoder", "BaseDecoder"]


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for model objects to ensure duck typing compatibility."""

    def forward(self, td: TensorDict, **kwargs: Any) -> TensorDict:
        """Forward pass signature."""
        ...

    def predict(self, td: TensorDict, **kwargs: Any) -> TensorDict:
        """Prediction signature."""
        ...


class BaseModel(nn.Module, ABC):
    """Abstract base class for all neural models in NGLab."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.cfg = kwargs

    @abstractmethod
    def forward(self, td: TensorDict, **kwargs: Any) -> TensorDict:
        """Standard forward pass for all NGLab models."""
        pass

    def predict(self, td: TensorDict, **kwargs: Any) -> TensorDict:
        """Default prediction logic (calls forward)."""
        return self.forward(td, **kwargs)

    def explain(self, td: TensorDict, **kwargs: Any) -> dict[str, Any]:
        """Default explanation logic (stub)."""
        return {"error": "Explanation not implemented for this model"}


class BaseEncoder(BaseModel, ABC):
    """Abstract base class for all encoders (e.g., VAE encoders)."""
    pass


class BaseDecoder(BaseModel, ABC):
    """Abstract base class for all decoders (e.g., VAE decoders)."""
    pass
