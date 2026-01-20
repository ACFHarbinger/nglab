"""
Denoising AutoEncoder (DAE) implementation.
"""

"""
Denoising AutoEncoder (DAE) implementation.
"""
import torch

from .ae import AutoEncoder


class DenoisingAE(AutoEncoder):
    """
    Denoising AutoEncoder (DAE).
    Adds Gaussian noise during training.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim,
        hidden_dims,
        latent_dim,
        noise_std=0.1,
        output_type="prediction",
    ):
        """Initialize Denoising AutoEncoder."""
        super().__init__(input_dim, hidden_dims, latent_dim, output_type)
        self.noise_std = noise_std

    def forward(self, x, return_embedding=None, return_sequence=False):
        """Forward pass."""
        if self.training:
            # Add noise
            noise = torch.randn_like(x) * self.noise_std
            x_noisy = x + noise
            return super().forward(x_noisy, return_embedding, return_sequence)
        else:
            return super().forward(x, return_embedding, return_sequence)
