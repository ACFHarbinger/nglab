"""
Rolling Price Window CNN for Time Series.
Inspired by "S&P 500 Stock's Movement Prediction using CNN".
"""

from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class RollingWindowCNN(nn.Module):
    """
    CNN for Time Series Prediction treating windows as 2D images.
    Input shape: (Batch, Seq_Len, Features)
    Internally reshaped to: (Batch, 1, Seq_Len, Features)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int = 30,
        hidden_dim: int = 64,
        output_type: str = "prediction",
    ) -> None:
        """
        Initialize the CNN.

        Args:
            input_dim (int): Number of input features (Feature dimension).
            output_dim (int): Output dimension.
            seq_len (int): Length of the time window.
            hidden_dim (int): Hidden dimension size for fully connected layer.
            output_type (str): 'prediction' or 'embedding'.
        """
        super().__init__()
        self.output_type = output_type

        # 1. Convolutional Layers
        # Treating time series as an image (Height=Seq_Len, Width=Features)
        # Or (Height=Features, Width=Seq_Len)?
        # Standard convention: (Batch, Channel, Height, Width)
        # Here: (Batch, 1, Seq_Len, Features) seems reasonable if Features are small.
        # But usually we convolute over Time.
        # Kernels often cover full feature width in 1D Conv.
        # The prompt specifies "Rolling Price Window CNN using 2D Convolutions".
        # This implies we scan over (Time, Features).

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)

        # Max Pooling to reduce dimensions
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        # (2, 1) pools time by 2, features kept? Or pool both?
        # Let's pool both for general reduction.
        self.pool_both = nn.MaxPool2d(kernel_size=(2, 2))

        # Calculate flattened dimension
        # Assuming 2 pools of (2,2) roughly divides dimensions by 4.
        # We'll compute dynamically in forward or assume fixed seq_len.

        # Let's use adaptive pooling to enforce a fixed output size before FC
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        # Output: (Batch, 32, 4, 4) -> 32*16 = 512 flat
        flat_dim = 32 * 4 * 4

        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, return_embedding: bool | None = None
    ) -> torch.Tensor:
        """
        Forward pass.
        x: (Batch, Seq_Len, Features)
        """
        # Add channel dimension: (B, 1, L, F)
        x_in = x.unsqueeze(1)

        # Block 1
        x_in = self.conv1(x_in)  # (B, 16, L, F)
        x_in = self.bn1(x_in)
        x_in = F.relu(x_in)
        x_in = self.pool_both(x_in)  # (B, 16, L/2, F/2)

        # Block 2
        x_in = self.conv2(x_in)  # (B, 32, L/2, F/2)
        x_in = self.bn2(x_in)
        x_in = F.relu(x_in)

        # Adaptive Pool to ensure fixed size regardless of input L or F slightly varying
        x_in = self.adaptive_pool(x_in)  # (B, 32, 4, 4)

        # Flatten
        x_flat = torch.flatten(x_in, 1)  # (B, 512)

        # FC
        x_emb = self.fc1(x_flat)  # (B, Hidden)
        x_emb = F.relu(x_emb)

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            return cast(torch.Tensor, x_emb)

        return cast(torch.Tensor, self.fc2(x_emb))
