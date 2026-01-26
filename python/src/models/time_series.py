from __future__ import annotations

from typing import Any, cast, TYPE_CHECKING

import torch

from .deep_factory import DEEP_MODEL_NAMES, create_deep_model
from .mac_factory import MAC_MODEL_NAMES, create_mac_model
from .base import BaseModel
from python.src.utils.registry import register_model

if TYPE_CHECKING:
    from tensordict import TensorDict


@register_model("TimeSeriesBackbone")
class TimeSeriesBackbone(BaseModel):
    """
    Unified Backbone for Time Series.
    Wraps specific implementations (Transformer, LSTM, etc).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """
        Initialize TimeSeriesBackbone.

        Args:
            cfg: Configuration dictionary defining the model architecture.
        """
        super().__init__()
        self.cfg = cfg
        model_name = cfg.get("name", "NSTransformer")

        # Try deep model factory first
        model = create_deep_model(model_name, cfg)

        # Try MAC model factory if not found
        if model is None:
            model = create_mac_model(model_name, cfg)

        # Raise error if still not found
        if model is None:
            raise ValueError(f"Unknown model: {model_name}")

        self.model = model

    def forward(self, td: TensorDict, **kwargs: Any) -> TensorDict:
        """
        Forward pass using TensorDict.

        Args:
            td: Input TensorDict containing 'observation'.
            kwargs: Additional arguments (e.g., 'return_sequence').

        Returns:
            TensorDict with 'state_value' or 'embedding'.
        """
        obs = td["observation"]

        # Whitelist for models supporting return_sequence
        sequence_supported = DEEP_MODEL_NAMES + MAC_MODEL_NAMES
        
        # Merge kwargs with cfg defaults
        forward_kwargs = {**kwargs}
        if self.cfg.get("return_sequence", False):
            forward_kwargs.setdefault("return_sequence", True)

        if self.cfg.get("name") not in sequence_supported:
            forward_kwargs.pop("return_sequence", None)

        out = self.model(obs, **forward_kwargs)
        
        # Return TensorDict
        td.set("state_value", out)
        return td

    @property
    def in_keys(self) -> list[str]:
        return ["observation"]

    @property
    def out_keys(self) -> list[str]:
        # If used as critic, output is value
        # If used as backbone, output is embedding
        # TorchRL might assume "state_value" for critic if not specified?
        return ["state_value"]
