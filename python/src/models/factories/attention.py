
from __future__ import annotations

from typing import Any

import torch.nn as nn

from python.src.models.attention.attention_net import AttentionNet
from python.src.models.attention.nstransformer import NSTransformer
from python.src.models.factories.base import NeuralComponentFactory


class AttentionFactory(NeuralComponentFactory):
    """Factory for attention mechanisms."""

    @staticmethod
    def get_component(name: str, **kwargs: Any) -> nn.Module:
        """Get attention model by name."""
        name = name.lower()
        # NSTransformer uses 'nstransformer' or 'transformer'
        if "nstransformer" in name:
            return NSTransformer(**kwargs)
        elif "attention" in name:
            return AttentionNet(**kwargs)
        else:
            raise ValueError(
                f"Unknown attention model: {name}. "
                f"Available: nstransformer, attention"
            )
