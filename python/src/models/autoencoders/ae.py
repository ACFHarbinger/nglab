"""
AutoEncoder (AE) implementation.
"""

from typing import cast

import torch
from torch import nn


class AutoEncoder(nn.Module):
    """
    Standard AutoEncoder (AE).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        output_type: str = "prediction",
    ) -> None:
        """Initialize AutoEncoder."""
        super().__init__()
        self.output_type = output_type

        # Encoder
        encoder_layers: list[nn.Module] = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            last_dim = h_dim
        encoder_layers.append(nn.Linear(last_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: list[nn.Module] = []
        last_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(last_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            last_dim = h_dim
        decoder_layers.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return cast(torch.Tensor, self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to input space."""
        return cast(torch.Tensor, self.decoder(z))

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass."""
        # Handle sequence
        z: torch.Tensor
        recon: torch.Tensor

        if x.dim() == 3:
            b, s, f = x.shape
            x_flat = x.view(b * s, f)
            z_flat = self.encode(x_flat)
            recon_flat = self.decode(z_flat)
            z = z_flat.view(b, s, -1)
            recon = recon_flat.view(b, s, -1)
        else:
            z = self.encode(x)
            recon = self.decode(z)

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            if not return_sequence and z.dim() == 3:
                return z[:, -1, :]
            return z

        if not return_sequence and recon.dim() == 3:
            return recon[:, -1, :]
        return recon
