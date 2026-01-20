"""
Denoising AutoEncoder (DAE) implementation.
"""

import torch
from typing import Optional

from .ae import AutoEncoder


class DenoisingAE(AutoEncoder):
    """
    Denoising AutoEncoder (DAE).
    Adds Gaussian noise during training.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        latent_dim,
        noise_std=0.1,
        output_type="prediction",
    ) -> None:
        """Initialize Denoising AutoEncoder."""
        super().__init__(input_dim, hidden_dims, latent_dim, output_type)
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor, return_embedding: Optional[bool] = None, return_sequence: bool = False) -> torch.Tensor:
        """Forward pass."""
        if self.training:
            # Add noise
            noise = torch.randn_like(x) * self.noise_std
            x_noisy = x + noise
            return super().forward(x_noisy, return_embedding, return_sequence)
        else:
            return super().forward(x, return_embedding, return_sequence)
