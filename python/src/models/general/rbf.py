"""
Radial Basis Function (RBF) Network.
"""

from typing import cast

import torch
from torch import nn


class RBF(nn.Module):
    """
    Radial Basis Function Network.
    Typically consist of an input layer, a hidden layer of RBF neurons, and a linear output layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_centers: int,
        output_dim: int,
        sigma: float = 1.0,
        output_type: str = "prediction",
    ) -> None:
        """
        Args:
            input_dim (int): Input feature dimension.
            num_centers (int): Number of RBF hidden units (centers).
            output_dim (int): Output dimension.
            sigma (float): Width of the radial basis function (Gaussian).
            output_type (str): 'prediction' or 'embedding'.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.output_dim = output_dim
        self.sigma = sigma
        self.output_type = output_type

        # Centers are learnable parameters
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))

        # Linear weights from hidden RBF to output
        self.linear = nn.Linear(num_centers, output_dim)

    def kernel_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gaussian RBF kernel: exp(-||x - c||^2 / (2 * sigma^2))
        x: (Batch, Input_Dim)
        """
        # (Batch, 1, Input_Dim) - (1, Num_Centers, Input_Dim)
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=2)
        return torch.exp(-dist_sq / (2 * self.sigma**2))

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass."""
        # Handle sequence input
        if x.dim() == 3:
            b, s, f = x.shape
            x_flat = x.view(b * s, f)
            emb = self.kernel_function(x_flat)
            emb = emb.view(b, s, -1)
        else:
            emb = self.kernel_function(x)

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            if not return_sequence and emb.dim() == 3:
                return emb[:, -1, :]
            return emb

        out = self.linear(emb)
        if not return_sequence and out.dim() == 3:
            return cast(torch.Tensor, out[:, -1, :])
        return cast(torch.Tensor, out)
