"""Activation functions wrapper for neural networks."""
import math
import torch
import torch.nn as nn

from typing import Optional, Tuple


class ActivationFunction(nn.Module):
    """
    Wrapper for various activation functions in PyTorch.
    
    Provides a unified interface to instantiate different activation functions
    by name, with support for parameters like threshold, negative slope, etc.
    """
    def __init__(self, 
                af_name: str='relu', 
                fparam: Optional[float]=None,
                tval: Optional[float]=None,
                rval: Optional[float]=None,
                n_params: Optional[int]=None,
                urange: Optional[Tuple[float, float]]=None,
                inplace: Optional[bool]=False):
        """
        Initializes the activation function.

        Args:
            af_name: Name of the activation function (e.g., 'relu', 'leakyrelu').
            fparam: Float parameter for activations (e.g., alpha for ELU, negative_slope for LeakyReLU).
            tval: Threshold value (used in Softshrink, Hardshrink, etc.).
            rval: Replacement value (used in global replacement logic if needed).
            n_params: Number of parameters for PReLU.
            urange: Uniform range for RReLU (lower, upper).
            inplace: Whether to perform the operation in-place.
        """
        super(ActivationFunction, self).__init__()
        if tval and rval is None and not af_name == 'softplus': 
            rval = tval # Replacement value = threshold
        if af_name == 'relu':
            self.activation = nn.ReLU(inplace=inplace)
        elif af_name == 'leakyrelu':
            self.activation = nn.LeakyReLU(inplace=inplace, negative_slope=fparam)
        elif af_name == 'silu':
            self.activation = nn.SiLU(inplace=inplace)
        elif af_name == 'selu':
            self.activation = nn.SELU(inplace=inplace)
        elif af_name == 'elu':
            self.activation = nn.ELU(inplace=inplace, alpha=fparam)
        elif af_name == 'celu':
            self.activation = nn.CELU(inplace=inplace, alpha=fparam)
        elif af_name == 'prelu':
            self.activation = nn.PReLU(num_parameters=n_params if n_params else 1, init=fparam if fparam else 0.25)
        elif af_name == 'rrelu':
            self.activation = nn.RReLU(inplace=inplace, lower=urange[0], upper=urange[1])
        elif af_name == 'gelu':
            self.activation = nn.GELU()
        elif af_name == 'gelu_tanh':
            self.activation = nn.GELU(approximate='tanh')
        elif af_name == 'tanh':
            self.activation = nn.Tanh()
        elif af_name == 'tanhshrink':
            self.activation = nn.Tanhshrink()
        elif af_name == 'mish':
            self.activation = nn.Mish(inplace=inplace)
        elif af_name == 'hardshrink':
            self.activation = nn.Hardshrink(lambd=fparam)
        elif af_name == 'hardtanh':
            self.activation = nn.Hardtanh(inplace=inplace, min_val=urange[0], max_val=urange[1])
        elif af_name == 'hardswish':
            self.activation = nn.Hardswish(inplace=inplace)
        elif af_name == 'glu':
            self.activation = nn.GLU(dim=fparam)
        elif af_name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif af_name == 'logsigmoid':
            self.activation = nn.LogSigmoid()
        elif af_name == 'hardsigmoid':
            self.activation = nn.Hardsigmoid(inplace=inplace)
        elif af_name == 'threshold':
            self.activation = nn.Threshold(inplace=inplace, threshold=tval, value=rval)
        elif af_name == 'softplus':
            self.activation = nn.Softplus(beta=fparam, threshold=tval)
        elif af_name == 'softshrink':
            self.activation = nn.Softshrink(lambd=fparam)
        elif af_name == 'softsign':
            self.activation = nn.Softsign()
        else:
            raise ValueError("Unknown activation function: {}".format(af_name))

        if isinstance(self.activation, nn.PReLU):
            self.init_parameters()

        if tval and rval and not isinstance(self.activation, nn.Threshold):
            self.tval = tval
            self.rval = rval
            self.apply_threshold = True
        else:
            self.apply_threshold = False

    def init_parameters(self):
        """Initializes the parameters of the activation function using uniform distribution."""
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        """
        Applies the activation function to the input.

        Args:
            input: Input tensor.
            mask: Optional mask (not used by default activations but kept for interface).

        Returns:
            Output tensor.
        """
        out = self.activation(input)
        if self.apply_threshold:
            out = torch.where(out > self.tval, self.rval, out)
        return out