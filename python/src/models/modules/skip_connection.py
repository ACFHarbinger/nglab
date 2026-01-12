"""Residual skip connection implementation."""
import torch.nn as nn


class SkipConnection(nn.Module):
    """
    Implements a residual connection: output = input + module(input).
    """
    def __init__(self, module:nn.Module):
        """
        Initializes the skip connection.

        Args:
            module: The neural network module to wrap with a residual connection.
        """
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, *args, **kwargs):
        """
        Applies the residual skip connection: input + module(input).

        Args:
            input: Input tensor.
            *args: Additional arguments for the module.
            **kwargs: Additional keyword arguments for the module.

        Returns:
            Result of the addition.
        """
        return input + self.module(input, *args, **kwargs)
