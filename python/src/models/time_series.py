"""
Unified Backbone for Time Series Models.
"""

from typing import Any, cast

import torch
from torch import nn

from .deep_factory import DEEP_MODEL_NAMES, create_deep_model
from .mac_factory import MAC_MODEL_NAMES, create_mac_model
from python.src.utils.registry import register_model


@register_model("TimeSeriesBackbone")
class TimeSeriesBackbone(nn.Module):
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

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor or dictionary containing 'observation'.

        Returns:
            Model output.
        """
        obs: torch.Tensor
        if isinstance(x, dict):
            obs = x["observation"]
        else:
            obs = x

        kwargs: dict[str, Any] = {}
        if self.cfg.get("return_sequence", False):
            kwargs["return_sequence"] = True

        # Whitelist for models supporting return_sequence
        sequence_supported = DEEP_MODEL_NAMES + MAC_MODEL_NAMES

        if self.cfg.get("name") in sequence_supported:
            pass
        elif "return_sequence" in kwargs:
            del kwargs["return_sequence"]

        out = self.model(obs, **kwargs)
        # We generally expect models to handle their return types.
        # Wrappers should return what's expected.
        return cast(torch.Tensor, out)

    @property
    def in_keys(self) -> list[str]:
        return ["observation"]

    @property
    def out_keys(self) -> list[str]:
        # If used as critic, output is value
        # If used as backbone, output is embedding
        # TorchRL might assume "state_value" for critic if not specified?
        return ["state_value"]
