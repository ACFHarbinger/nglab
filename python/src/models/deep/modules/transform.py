"""
Transformation layers for model dimensions.
"""

import torch


class Transpose(torch.nn.Module):
    """
    Transpose layer for neural networks.
    """

    def __init__(self, dims: tuple[int, int] = (-1, 1)) -> None:
        """
        Initialize.

        Args:
            dims (tuple): Dimensions to transpose.
        """
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        return torch.transpose(x, *self.dims)
