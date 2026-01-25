"""
Deep Belief Network (DBN) implementation.
"""

from typing import Any

import torch
from torch import nn

from .rbm import RBM


class DeepBeliefNetwork(nn.Module):
    """
    Deep Belief Network (DBN) - Stack of RBMs trained layer-by-layer.
    Greedy layer-wise pretraining; forward/backward passes for encoding/decoding.
    """

    def __init__(self, layer_sizes: list[int], output_type: str = "prediction") -> None:
        """
        Args:
            layer_sizes: [input_dim, h1, h2, ..., latent_dim]
        """
        super().__init__()
        self.layer_sizes = layer_sizes
        self.output_type = output_type

        self.rbms = nn.ModuleList(
            [
                RBM(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass."""
        # Forward pass through the stack of RBMs
        current = x
        for rbm in self.rbms:
            # We use the sigmoid probabilities for deeper layers during forward pass
            assert isinstance(rbm, RBM)
            current, _ = rbm.sample_h(current)

        if not return_sequence and current.ndim == 3:
            return current[:, -1, :]
        return current

    def pretrain(self, data_loader: Any, epochs: int = 10) -> None:
        """Example placeholder for greedy layer-wise pretraining."""
        # Typically one would train each RBM in sequence using CD-k
        pass
