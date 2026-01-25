"""
Multi-Layer Perceptron (MLP) implementation.
"""

from typing import cast

import torch
from torch import nn


class MLP(nn.Module):
    """
    Multi-layer Perceptron (FF / DFF).
    Supports a configurable number of hidden layers and activation functions.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
        output_type: str = "prediction",
    ) -> None:
        """
        Args:
            input_dim (int): Dimension of input features.
            hidden_dims (list): List of hidden layer dimensions.
            output_dim (int): Dimension of output.
            dropout (float): Dropout probability.
            activation (str): Activation function name ('relu', 'tanh', 'gelu', 'sigmoid').
            output_type (str): 'prediction' or 'embedding'.
        """
        super().__init__()
        self.output_type = output_type

        layers: list[nn.Module] = []
        last_dim = input_dim

        # Select activation
        act_fn_cls: type[nn.Module] = cast(
            type[nn.Module],
            {
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "gelu": nn.GELU,
                "sigmoid": nn.Sigmoid,
            }.get(activation.lower(), nn.ReLU),
        )

        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(act_fn_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        x: (Batch, Features) or (Batch, Seq, Features)
        """
        emb: torch.Tensor
        if x.dim() == 3:
            # Apply per time step or flatten?
            # Existing backbone pattern usually handles per-step in RNNs,
            # but MLPs usually flatten or apply to last step.
            # Let's apply to all steps if 3D.
            b, s, f = x.shape
            x_flat = x.view(b * s, f)
            emb_flat = cast(torch.Tensor, self.backbone(x_flat))
            emb = emb_flat.view(b, s, -1)
        else:
            emb = cast(torch.Tensor, self.backbone(x))

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            if not return_sequence and emb.dim() == 3:
                return emb[:, -1, :]
            return emb

        out = cast(torch.Tensor, self.head(emb))
        if not return_sequence and out.dim() == 3:
            return out[:, -1, :]
        return out
