"""
Self-Organizing Map (SOM) implementation.
"""

import torch
from torch import nn


class KohonenMap(nn.Module):
    """
    Kohonen Self-Organizing Map (SOM).
    """

    def __init__(
        self,
        input_dim: int,
        grid_size: tuple[int, int] = (10, 10),
        output_type: str = "embedding",
    ) -> None:
        """Initialize SOM."""
        super().__init__()
        self.input_dim = input_dim
        self.grid_size = grid_size
        self.num_neurons = grid_size[0] * grid_size[1]
        self.output_type = output_type

        self.weights = nn.Parameter(torch.rand(self.num_neurons, input_dim))

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass - find BMU."""
        if x.dim() == 3:
            b, s, f = x.shape
            x_flat = x.view(b * s, f)
            bmu = self._find_bmu(x_flat)
            bmu.view(b, s, -1)  # Wait, BMU is index.
            # Usually for SOM we might return the weights of BMU as embedding
            emb = self.weights[bmu].view(b, s, -1)
        else:
            bmu = self._find_bmu(x)
            emb = self.weights[bmu]

        # For SOM, we'll return the BMU weights as embedding
        if not return_sequence and emb.dim() == 3:
            return emb[:, -1, :]
        return emb

    def _find_bmu(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.weights.unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=2)
        return torch.argmin(dist_sq, dim=1)
