"""Normalized activation functions (Softmax, etc.) with adaptive options."""
import math
import torch.nn as nn

from typing import Optional, Sequence


class NormalizedActivationFunction(nn.Module):
    """
    Wrapper for normalized activation functions (Softmax, LogSoftmax, etc.).
    """
    def __init__(self, naf_name: str='softmax', dim: Optional[int]=-1, n_classes: Optional[int]=None, cutoffs: Optional[Sequence[int]]=None, dval: Optional[float]=4.0, bias: Optional[bool]=False):
        """
        Initializes the normalized activation function.

        Args:
            naf_name: Name of the normalized activation function ('softmax', 'logsoftmax', etc.).
            dim: Dimension along which the activation is applied.
            n_classes: Number of classes (for adaptive softmax).
            cutoffs: Cutoffs for adaptive softmax clusters.
            dval: Divisor value for adaptive softmax.
            bias: Whether to use bias in adaptive softmax.
        """
        super(NormalizedActivationFunction, self).__init__()
        if naf_name == 'softmin':
            self.norm_activation = nn.Softmin(dim=dim)
        elif naf_name == 'softmax':
            self.norm_activation = nn.Softmax(dim=dim)
        elif naf_name == 'logsoftmax':
            self.norm_activation = nn.LogSoftmax(dim=dim)
        elif naf_name == 'softmax2d':
            self.norm_activation = nn.Softmax2d()
        elif naf_name == 'adaptivelogsoftmax':
            if n_classes is None:
                raise ValueError("n_classes must be provided for adaptivelogsoftmax")
            # Ensure cutoffs is list
            if cutoffs is None:
                 cutoffs = [n_classes // 4, n_classes // 2, 3 * n_classes // 4]
            self.norm_activation = nn.AdaptiveLogSoftmaxWithLoss(in_features=dim, n_classes=n_classes, cutoffs=cutoffs, div_value=dval, head_bias=bias)
        else:
            raise ValueError("Unknown normalized activation function: {}".format(naf_name))

        if isinstance(self.norm_activation, nn.AdaptiveLogSoftmaxWithLoss):
            self.init_parameters()

    def init_parameters(self):
        """Initializes the parameters if applicable."""
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        """
        Applies the normalized activation function to the input.

        Args:
            input: Input tensor.
            mask: Optional mask (not used by all activations).

        Returns:
            Output tensor.
        """
        return self.norm_activation(input)