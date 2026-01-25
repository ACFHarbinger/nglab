
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class NeuralComponentFactory(ABC):
    """Abstract base class for neural component factories."""

    @staticmethod
    @abstractmethod
    def get_component(name: str, **kwargs: Any) -> nn.Module:
        """Create a neural component instance."""
        pass
