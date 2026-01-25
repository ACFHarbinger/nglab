"""
Deep Convolutional Network (DCN).
"""

from typing import cast

import torch
from torch import nn


class DeepConvNet(nn.Module):
    """
    Deep Convolutional Network (DCN) - Multi-layer CNN for hierarchical feature learning.
    Batch normalization and pooling layers; feature extraction at different layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: list[int] | None = None,
        output_dim: int = 1,
        output_type: str = "prediction",
    ) -> None:
        """Initialize DCN."""
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_type = output_type

        layers: list[nn.Module] = []

        # Handle input_dim if we treat it as (Channels, Seq) or similar
        # For simplicity, we assume input is (Batch, Seq, Features) and we treat Features as channels

        last_channels = input_dim
        for h_channels in hidden_channels:
            layers.append(
                nn.Conv1d(last_channels, h_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.BatchNorm1d(h_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            last_channels = h_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(last_channels, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        x: (Batch, Seq, Features) -> (Batch, Features, Seq)
        """
        # Conv1d expects (Batch, Channels, Length)
        x_in = x.transpose(1, 2)

        feat = cast(torch.Tensor, self.features(x_in))
        pooled = cast(torch.Tensor, self.pool(feat)).squeeze(-1)

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            return pooled

        return cast(torch.Tensor, self.classifier(pooled))
