
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class ModelBase(nn.Module, ABC):
    """Base class for all models."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass logic."""


class BaseEncoder(ModelBase):
    """Base encoder interface."""


class BaseDecoder(ModelBase):
    """Base decoder interface."""
