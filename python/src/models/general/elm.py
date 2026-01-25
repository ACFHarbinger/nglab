"""
Extreme Learning Machine (ELM) implementation.
"""

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class ELM(nn.Module):
    """
    Extreme Learning Machine (ELM).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "sigmoid",
        output_type: str = "prediction",
    ) -> None:
        """Initialize ELM."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_type = output_type

        # These are usually not trained in ELM
        self.register_buffer("w", torch.randn(hidden_dim, input_dim))
        self.register_buffer("b", torch.randn(hidden_dim))

        self.act_fn: Any = {
            "sigmoid": torch.sigmoid,
            "relu": torch.relu,
            "tanh": torch.tanh,
        }.get(activation.lower(), torch.sigmoid)

        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass."""
        h: torch.Tensor
        out: torch.Tensor

        w = self.w if isinstance(self.w, torch.Tensor) else torch.tensor(self.w)
        b = self.b if isinstance(self.b, torch.Tensor) else torch.tensor(self.b)

        if x.dim() == 3:
            batch_size, seq_len, feat_dim = x.shape
            x_flat = x.view(batch_size * seq_len, feat_dim)
            h_flat = self.act_fn(F.linear(x_flat, w, b))
            out_flat = self.readout(h_flat)
            h = h_flat.view(batch_size, seq_len, -1)
            out = out_flat.view(batch_size, seq_len, -1)
        else:
            h = self.act_fn(F.linear(x, w, b))
            out = self.readout(h)

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        res = h if should_return_embedding else out

        if not return_sequence and res.dim() == 3:
            return res[:, -1, :]
        return res
