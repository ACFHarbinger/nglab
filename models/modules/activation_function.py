import math
import torch
import torch.nn as nn

from typing import Optional, Tuple


class ActivationFunction(nn.Module):
    def __init__(self, 
                af_name: str='relu', 
                fparam: Optional[float]=None,
                tval: Optional[float]=None,
                rval: Optional[float]=None,
                n_params: Optional[int]=None,
                urange: Optional[Tuple[float, float]]=None,
                inplace: Optional[bool]=False):
        super(ActivationFunction, self).__init__()
        if tval and rval is None and not af_name == 'softplus': rval = tval # Replacement value = threshold
        self.activation = {
            'relu': nn.ReLU(inplace=inplace),
            'leakyrelu': nn.LeakyReLU(inplace=inplace, negative_slope=fparam),
            'silu': nn.SiLU(inplace=inplace),
            'selu': nn.SELU(inplace=inplace),
            'elu': nn.ELU(inplace=inplace, alpha=fparam),
            'celu': nn.CELU(inplace=inplace, alpha=fparam),
            'prelu': nn.PReLU(num_parameters=n_params, init=fparam), # weight decay should not be used when learning $a$ for good performance.
            'rrelu': nn.RReLU(inplace=inplace, lower=urange[0], upper=urange[1]),
            'gelu': nn.GELU(),
            'gelu_tanh': nn.GELU(approximate='tanh'),
            'tanh': nn.Tanh(),
            'tanhshrink': nn.Tanhshrink(),
            'mish': nn.Mish(inplace=inplace),
            'hardshrink': nn.Hardshrink(lambd=fparam),
            'hardtanh': nn.Hardtanh(inplace=inplace, min_val=urange[0], max_val=urange[1]),
            'hardswish': nn.Hardswish(inplace=inplace),
            'glu': nn.GLU(dim=fparam),
            'sigmoid': nn.Sigmoid(),
            'logsigmoid': nn.LogSigmoid(),
            'hardsigmoid': nn.Hardsigmoid(inplace=inplace),
            'threshold': nn.Threshold(inplace=inplace, threshold=tval, value=rval),
            'softplus': nn.Softplus(beta=fparam, threshold=tval),
            'softshrink': nn.Softshrink(lambd=fparam),
            'softsign': nn.Softsign()
        }.get(af_name, None)
        assert self.activation is not None, "Unknown activation function: {}".format(af_name)

        if isinstance(self.activation, nn.PReLU):
            self.init_parameters()

        if tval and rval and not isinstance(self.activation, nn.Threshold):
            self.activation = lambda x: torch.where(self.activation(x) > tval, rval, self.activation(x))

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return self.activation(input)