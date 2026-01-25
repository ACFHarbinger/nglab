"""
Deep Residual Network (ResNet) - Network with skip connections
"""

from typing import Literal, cast

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    Basic Residual Block with skip connection.

    Implements: output = activation(F(x) + x)
    where F(x) is a sequence of linear/conv layers.

    Args:
        hidden_dim: Dimension of the block
        use_conv: If True, use Conv1d; otherwise use Linear
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int,
        use_conv: bool = False,
        dropout: float = 0.1,
        kernel_size: int = 3,
    ) -> None:
        """Initialize Residual Block."""
        super().__init__()
        self.use_conv = use_conv

        if use_conv:
            self.block = nn.Sequential(
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(hidden_dim),
            )
        else:
            self.block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection.

        Args:
            x: Input tensor

        Returns:
            Output with skip connection applied
        """
        residual = x
        out = cast(torch.Tensor, self.block(x))
        out = out + residual  # Skip connection
        out = cast(torch.Tensor, self.activation(out))
        return out


class DeepResNet(nn.Module):
    """
    Deep Residual Network (ResNet) for time series or sequence data.

    ResNet uses skip connections to enable training of very deep networks
    by addressing the vanishing gradient problem.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for residual blocks
        num_blocks: Number of residual blocks
        output_dim: Output dimension
        use_conv: If True, use convolutional blocks; otherwise use fully connected
        dropout: Dropout probability
        output_type: 'prediction' returns final output, 'embedding' returns features
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        output_dim: int = 10,
        use_conv: bool = False,
        dropout: float = 0.1,
        kernel_size: int = 3,
        output_type: Literal["prediction", "embedding"] = "prediction",
    ) -> None:
        """
        Initialize DeepResNet.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_conv = use_conv
        self.output_type = output_type

        # Initial projection to hidden_dim
        if use_conv:
            self.input_proj = nn.Sequential(
                nn.Conv1d(
                    input_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
            )

        # Stack of residual blocks
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    hidden_dim,
                    use_conv=use_conv,
                    dropout=dropout,
                    kernel_size=kernel_size,
                )
                for _ in range(num_blocks)
            ]
        )

        # Global pooling for conv case
        if use_conv:
            self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        Forward pass through Deep ResNet.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            return_sequence: If True, return output for each timestep

        Returns:
            Output tensor of shape (batch_size, output_dim) or (batch_size, seq_len, output_dim)
        """
        # Handle input shape
        if self.use_conv:
            if x.dim() == 3:
                x = x.transpose(1, 2)  # (batch, input_dim, seq)
            elif x.dim() == 2:
                x = x.unsqueeze(-1)
        else:
            # For non-conv, keep (batch, seq, features)
            pass

        # Initial projection
        x = cast(torch.Tensor, self.input_proj(x))

        # Pass through residual blocks
        for block in self.res_blocks:
            x = cast(torch.Tensor, block(x))

        if self.output_type == "embedding":
            if self.use_conv:
                if return_sequence:
                    return x.transpose(1, 2)
                else:
                    x = cast(torch.Tensor, self.global_pool(x)).squeeze(-1)
                    return x
            elif return_sequence:
                return x
            else:
                return x[:, -1, :]

        # Apply output projection
        out: torch.Tensor
        if self.use_conv:
            if return_sequence:
                # Transpose to (batch, seq, hidden_dim) for output projection
                x = x.transpose(1, 2)
                out = cast(torch.Tensor, self.output_proj(x))
            else:
                # Global pooling
                x = cast(torch.Tensor, self.global_pool(x)).squeeze(-1)
                out = cast(torch.Tensor, self.output_proj(x))
        # Fully connected case
        elif return_sequence:
            out = cast(torch.Tensor, self.output_proj(x))
        else:
            out = cast(torch.Tensor, self.output_proj(x[:, -1, :]))

        return out


class ResNetBottleneck(nn.Module):
    """
    Bottleneck Residual Block (used in deeper ResNets like ResNet-50).

    Uses 1x1 conv to reduce dimensions, 3x3 conv for computation,
    then 1x1 conv to restore dimensions.

    Args:
        in_dim: Input dimension
        bottleneck_dim: Bottleneck dimension (typically in_dim // 4)
        out_dim: Output dimension
        use_conv: If True, use Conv1d; otherwise use Linear
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_dim: int,
        bottleneck_dim: int,
        out_dim: int,
        use_conv: bool = False,
        dropout: float = 0.1,
    ) -> None:
        """Initialize Bottleneck Block."""
        super().__init__()
        self.use_conv = use_conv
        self.shortcut: nn.Module

        if use_conv:
            self.block = nn.Sequential(
                nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(),
                nn.Conv1d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(bottleneck_dim, out_dim, kernel_size=1),
                nn.BatchNorm1d(out_dim),
            )

            # Shortcut connection (if dimensions don't match)
            if in_dim != out_dim:
                self.shortcut = nn.Conv1d(in_dim, out_dim, kernel_size=1)
            else:
                self.shortcut = nn.Identity()
        else:
            self.block = nn.Sequential(
                nn.Linear(in_dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, out_dim),
                nn.LayerNorm(out_dim),
            )

            if in_dim != out_dim:
                self.shortcut = nn.Linear(in_dim, out_dim)
            else:
                self.shortcut = nn.Identity()

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with bottleneck skip connection."""
        residual = cast(torch.Tensor, self.shortcut(x))
        out = cast(torch.Tensor, self.block(x))
        out = out + residual
        out = cast(torch.Tensor, self.activation(out))
        return out
