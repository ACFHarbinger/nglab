"""
Sparse AutoEncoder (SAE) implementation.
"""

import torch

from .ae import AutoEncoder


class SparseAE(AutoEncoder):
    """
    Sparse AutoEncoder (SAE).
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        sparsity_target: float = 0.05,
        sparsity_weight: float = 0.1,
        output_type: str = "prediction",
    ) -> None:
        """Initialize SAE."""
        super().__init__(input_dim, hidden_dims, latent_dim, output_type)
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass."""
        # Standard AE forward logic, but we might want to return activations for loss
        # Since this is a backbone class, we'll keep standard signatures.
        return super().forward(x, return_embedding, return_sequence)

    def kl_divergence(self, rho: float, rho_hat: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence."""
        rho_hat = torch.clamp(rho_hat, 1e-6, 1.0 - 1e-6)
        return rho * torch.log(torch.tensor(rho) / rho_hat) + (1 - rho) * torch.log(
            torch.tensor(1 - rho) / (1 - rho_hat)
        )

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute sparsity loss."""
        rho_hat = torch.mean(torch.sigmoid(z), dim=0)
        kl = self.kl_divergence(self.sparsity_target, rho_hat)
        return torch.sum(kl) * self.sparsity_weight
