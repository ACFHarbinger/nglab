"""
Transformation layers for model dimensions.
"""
import torch


class Transpose(torch.nn.Module):
    """
    Transpose layer for neural networks.
    """
    def __init__(self, dims=(-1, 1)):
        """
        Initialize.

        Args:
            dims (tuple): Dimensions to transpose.
        """
        super(Transpose, self).__init__()
        self.dims = dims

    def forward(self, x):
        """
        Forward pass.
        """
        return torch.transpose(x, *self.dims)