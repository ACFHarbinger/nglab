"""
Restricted Boltzmann Machine (RBM).
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class RBM(nn.Module):
    """
    Restricted Boltzmann Machine (RBM).
    """

    def __init__(
        self, visible_dim: int, hidden_dim: int, output_type: str = "embedding"
    ) -> None:
        """Initialize RBM."""
        super().__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.output_type = output_type

        self.weights = nn.Parameter(torch.randn(hidden_dim, visible_dim) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(visible_dim))
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim))

    def sample_h(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample hidden states given visible states."""
        prob = torch.sigmoid(F.linear(v, self.weights, self.h_bias))
        return prob, torch.bernoulli(prob)

    def sample_v(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample visible states given hidden states."""
        prob = torch.sigmoid(F.linear(h, self.weights.t(), self.v_bias))
        return prob, torch.bernoulli(prob)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass - Gibbs sampling step."""
        # x is visible layer v
        if x.dim() == 3:
            b, s, f = x.shape
            v_flat = x.view(b * s, f)
            _, h = self.sample_h(v_flat)
            _, v_recon = self.sample_v(h)
            h = h.view(b, s, -1)
            v_recon = v_recon.view(b, s, -1)
        else:
            _p_h, h = self.sample_h(x)
            _p_v, v_recon = self.sample_v(h)

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            # Return hidden activations as embedding
            out = h
        else:
            # Return reconstruction
            out = v_recon

        if not return_sequence and out.dim() == 3:
            return out[:, -1, :]
        return out
