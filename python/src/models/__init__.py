from __future__ import annotations

from python.src.models.base import BaseModel, BaseEncoder, BaseDecoder
from python.src.models.registry import MODEL_REGISTRY, get_model, register_model
from python.src.models.time_series import TimeSeriesBackbone

# Flattened Architecture Modules
from .attention import *  # noqa: F403
from .autoencoders import *  # noqa: F403
from .competitive import *  # noqa: F403
from .convolutional import *  # noqa: F403
from .general import *  # noqa: F403
from .memory import *  # noqa: F403
from .probabilistic import *  # noqa: F403
from .recurrent import *  # noqa: F403
from .spiking import *  # noqa: F403

__all__ = [
    "BaseModel",
    "BaseEncoder",
    "BaseDecoder",
    "MODEL_REGISTRY",
    "get_model",
    "register_model",
    "TimeSeriesBackbone",
]
