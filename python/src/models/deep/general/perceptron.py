"""
Perceptron implementation.
"""

from typing import Any, cast

import torch
from torch import nn


class Perceptron(nn.Module):
    """
    Perceptron (P) - Basic single-layer feedforward network.
    The simplest neural network with configurable activation functions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "sigmoid",
        output_type: str = "prediction",
    ) -> None:
        """Initialize Perceptron."""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_type = output_type

        self.linear = nn.Linear(input_dim, output_dim)

        self.act_fn: Any = {
            "sigmoid": torch.sigmoid,
            "relu": torch.relu,
            "tanh": torch.tanh,
            "step": lambda x: (x > 0).float(),
        }.get(activation.lower(), torch.sigmoid)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass."""
        # Handle sequence
        if x.dim() == 3:
            b, s, f = x.shape
            x_flat = x.view(b * s, f)
            out = self.act_fn(self.linear(x_flat))
            res = out.view(b, s, -1)
        else:
            res = self.act_fn(self.linear(x))

        if not return_sequence and res.dim() == 3:
            return cast(torch.Tensor, res[:, -1, :].clone())
        return cast(torch.Tensor, res)
