"""
Hopfield Network implementation.
"""

import torch
from torch import nn


class HopfieldNetwork(nn.Module):
    """
    Discrete Hopfield Network.
    """

    def __init__(self, size: int, output_type: str = "embedding") -> None:
        """Initialize Hopfield Network."""
        super().__init__()
        self.size = size
        self.output_type = output_type
        self.register_buffer("weights", torch.zeros(size, size))

    def store_patterns(self, patterns: torch.Tensor) -> None:
        """Store patterns using Hebbian learning."""
        w = torch.matmul(patterns.t(), patterns) / self.size
        w.fill_diagonal_(0)
        self.weights = w

    def forward(
        self,
        x: torch.Tensor,
        iterations: int = 10,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Retrieval as a 'forward' pass.
        """
        # Handle sequence
        if x.dim() == 3:
            b, s, f = x.shape
            x_flat = x.view(b * s, f)
            y_flat = self._retrieve(x_flat, iterations)
            out = y_flat.view(b, s, -1)
        else:
            out = self._retrieve(x, iterations)

        if not return_sequence and out.dim() == 3:
            return out[:, -1, :]
        return out

    def _retrieve(self, x: torch.Tensor, iterations: int) -> torch.Tensor:
        """Perform retrieval by iteratively updating the state."""
        s = x
        for _ in range(iterations):
            s = torch.sign(torch.matmul(s, self.weights))
            s[s == 0] = 1
        return s
