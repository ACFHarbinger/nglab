import math
import torch.nn as nn

from typing import Optional, Sequence


class NormalizedActivationFunction(nn.Module):
    def __init__(self, 
                naf_name: str='softmax', 
                dim: Optional[int]=-1, 
                n_classes: Optional[int]=None, 
                cutoffs: Optional[Sequence[int]]=None, 
                dval: Optional[float]=4.0, 
                bias: Optional[bool]=False):
        super(NormalizedActivationFunction, self).__init__()
        self.norm_activation = {
            'softmin': nn.Softmin(dim=dim), 
            'softmax': nn.Softmax(dim=dim),
            'logsoftmax': nn.LogSoftmax(dim=dim),
            'softmax2d': nn.Softmax2d(), 
            'adaptivelogsoftmax': nn.AdaptiveLogSoftmaxWithLoss(in_features=dim, n_classes=n_classes, cutoffs=cutoffs, div_value=dval, head_bias=bias)
        }.get(naf_name, None)
        assert self.norm_activation is not None, "Unknown normalized activation function: {}".format(naf_name)

        if isinstance(self.norm_activation, nn.AdaptiveLogSoftmaxWithLoss):
            self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return self.norm_activation(input)