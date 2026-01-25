"""
Deep Convolutional Inverse Graphics Network (DCIGN).
"""

from typing import Any, cast

import torch
from torch import nn


class DCIGN(nn.Module):
    """
    Deep Convolutional Inverse Graphics Network (DCIGN).
    Disentangled representation learning; separates intrinsic from extrinsic features.
    Supports property swapping and manipulation.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_channels: list[int] | None = None,
        num_intrinsic: int = 32,
        num_extrinsic: int = 96,
        output_type: str = "prediction",
    ) -> None:
        """Initialize DCIGN."""
        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_intrinsic = num_intrinsic  # e.g., identity, shape
        self.num_extrinsic = num_extrinsic  # e.g., pose, lighting
        self.output_type = output_type

        # Encoder (Hierarchical feature learning)
        enc_layers: list[nn.Module] = []
        last_channels = input_dim
        for h_channels in hidden_channels:
            enc_layers.append(
                nn.Conv1d(last_channels, h_channels, kernel_size=4, stride=2, padding=1)
            )
            enc_layers.append(nn.BatchNorm1d(h_channels))
            enc_layers.append(nn.ReLU())
            last_channels = h_channels

        self.encoder = nn.Sequential(*enc_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_latent = nn.Linear(last_channels, latent_dim)

        # Decoder (Generation)
        dec_layers: list[nn.Module] = []
        last_channels = latent_dim
        for h_channels in reversed(hidden_channels):
            dec_layers.append(
                nn.ConvTranspose1d(
                    last_channels, h_channels, kernel_size=4, stride=2, padding=1
                )
            )
            dec_layers.append(nn.BatchNorm1d(h_channels))
            dec_layers.append(nn.ReLU())
            last_channels = h_channels

        self.decoder_body = nn.Sequential(*dec_layers)
        self.final_conv = nn.Conv1d(last_channels, input_dim, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input into intrinsic and extrinsic features."""
        # x: (Batch, Seq, Input_Dim) -> (Batch, Input_Dim, Seq)
        x_in = x.transpose(1, 2)
        feat = cast(torch.Tensor, self.encoder(x_in))
        z = cast(torch.Tensor, self.fc_latent(self.pool(feat).squeeze(-1)))

        # Split latent vector into intrinsic and extrinsic
        intrinsic = z[:, : self.num_intrinsic]
        extrinsic = z[:, self.num_intrinsic :]
        return intrinsic, extrinsic

    def decode(self, intrinsic: torch.Tensor, extrinsic: torch.Tensor) -> torch.Tensor:
        """Decode intrinsic and extrinsic features to reconstruct input."""
        z = torch.cat([intrinsic, extrinsic], dim=1)
        z = z.unsqueeze(-1)  # (Batch, Latent, 1)
        recon = cast(torch.Tensor, self.decoder_body(z))
        recon = cast(torch.Tensor, self.final_conv(recon))
        return recon.transpose(1, 2)  # (Batch, Seq_out, Input_Dim)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass."""
        intrinsic, extrinsic = self.encode(x)

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            return torch.cat([intrinsic, extrinsic], dim=1)

        recon = self.decode(intrinsic, extrinsic)
        if not return_sequence:
            return recon[:, -1, :]
        return recon
