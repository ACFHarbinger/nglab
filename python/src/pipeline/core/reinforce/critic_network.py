"""
Critic Network for REINFORCE with Baseline.

Estimates the value of a given state to provide a baseline for advantage
calculation in Reinforcement Learning.
"""

from typing import Any, cast

import torch
from torch import nn

from ..models.graph_encoder import GraphAttentionEncoder


class CriticNetwork(nn.Module):
    """
    Critic network based on Graph Attention Encoder and a Linear Value Head.
    """

    def __init__(  # noqa: PLR0913
        self,
        method: Any,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
        encoder_normalization: Any,
    ) -> None:
        """
        Initialize the critic network.
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        self.encoder = GraphAttentionEncoder(
            method=method,
            node_dim=input_dim,
            n_heads=8,
            embed_dim=embedding_dim,
            n_layers=n_layers,
            normalization=encoder_normalization,
        )

        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the critic network.

        Args:
            inputs: (batch_size, graph_size, input_dim) - Input feature tensors.

        Returns:
            torch.Tensor: Estimated state values.
        """
        _, graph_embeddings = self.encoder(inputs)
        return cast(torch.Tensor, self.value_head(graph_embeddings))
