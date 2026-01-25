"""
Deconvolutional Network (DN) implementation.
"""

from typing import Any, cast

import torch
from torch import nn


class DeconvNet(nn.Module):
    """
    Deconvolutional Network (DN) - Transposed convolutions for upsampling.
    Used for generation and reconstruction tasks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: list[int] | None = None,
        output_dim: int = 1,
        output_type: str = "prediction",
    ) -> None:
        """Initialize Deconvolutional Network."""
        if hidden_channels is None:
            hidden_channels = [128, 64, 32]
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_type = output_type

        layers: list[nn.Module] = []
        last_channels = input_dim

        for h_channels in hidden_channels:
            # Transposed conv for upsampling
            layers.append(
                nn.ConvTranspose1d(
                    last_channels, h_channels, kernel_size=4, stride=2, padding=1
                )
            )
            layers.append(nn.BatchNorm1d(h_channels))
            layers.append(nn.ReLU())
            last_channels = h_channels

        self.decoder = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(last_channels, output_dim, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        x: (Batch, Input_Dim) (latent/bottleneck) or (Batch, Units, Seq)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (Batch, Input_Dim, 1)

        out = cast(torch.Tensor, self.decoder(x))
        out = cast(torch.Tensor, self.final_conv(out))

        # (Batch, Output_Dim, Seq_out) -> (Batch, Seq_out, Output_Dim)
        out = out.transpose(1, 2)

        if not return_sequence:
            return out[:, -1, :]
        return out


class AutoDeconvNet(nn.Module):
    """
    AutoDeconvNet - Uses DeconvNet for autoencoder architecture.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_channels: list[int] | None = None,
        output_type: str = "prediction",
    ) -> None:
        """Initialize AutoDeconvNet."""
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_type = output_type

        # Encoder (Mirror of DeconvNet)
        layers: list[nn.Module] = []
        last_channels = input_dim
        for h_channels in hidden_channels:
            layers.append(
                nn.Conv1d(last_channels, h_channels, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.BatchNorm1d(h_channels))
            layers.append(nn.ReLU())
            last_channels = h_channels

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_latent = nn.Linear(last_channels, latent_dim)

        # Decoder
        self.decoder: DeconvNet = DeconvNet(
            latent_dim, list(reversed(hidden_channels)), input_dim, output_type
        )

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass."""
        # x: (Batch, Seq, Input_Dim) -> (Batch, Input_Dim, Seq)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x_in = x.transpose(1, 2)

        e = cast(torch.Tensor, self.encoder(x_in))
        z = cast(torch.Tensor, self.fc_latent(self.pool(e).squeeze(-1)))

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            return z

        return cast(torch.Tensor, self.decoder(z, return_sequence=return_sequence))
