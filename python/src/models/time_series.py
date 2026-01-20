"""
Unified Backbone for Time Series Models.
"""

from torch import nn

from .deep_factory import DEEP_MODEL_NAMES, create_deep_model
from .mac_factory import MAC_MODEL_NAMES, create_mac_model


class TimeSeriesBackbone(nn.Module):
    """
    Unified Backbone for Time Series.
    Wraps specific implementations (Transformer, LSTM, etc).
    """

    def __init__(self, cfg):
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

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor or dictionary containing 'observation'.

        Returns:
            Model output.
        """
        if hasattr(x, "get"):
            x = x.get("observation")

        kwargs = {}
        if self.cfg.get("return_sequence", False):
            kwargs["return_sequence"] = True

        # Whitelist for models supporting return_sequence
        sequence_supported = DEEP_MODEL_NAMES + MAC_MODEL_NAMES

        if self.cfg.get("name") in sequence_supported:
            pass
        elif "return_sequence" in kwargs:
            del kwargs["return_sequence"]

        out = self.model(x, **kwargs)
        return out
