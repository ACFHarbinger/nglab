import math
import torch.nn as nn

from typing import Optional


class Normalization(nn.Module):
    def __init__(self, 
                embed_dim: int, 
                norm_name: str='batch', 
                eps_alpha: float=1e-05, 
                learn_affine: Optional[bool]=True, 
                track_stats: Optional[bool]=False,  
                mbval: Optional[float]=None, 
                n_groups: Optional[int]=None, 
                kval: Optional[float]=None,
                bias: Optional[bool]=True):
        super(Normalization, self).__init__()
        self.normalizer = {
            'instance': nn.InstanceNorm1d(embed_dim, eps=eps_alpha, affine=learn_affine, track_running_stats=track_stats, momentum=mbval),
            'batch': nn.BatchNorm1d(embed_dim, eps=eps_alpha, affine=learn_affine, track_running_stats=track_stats, momentum=mbval),
            'layer': nn.LayerNorm(embed_dim, eps=eps_alpha, elementwise_affine=learn_affine, bias=bias),
            'group': nn.GroupNorm(n_groups, eps=eps_alpha, num_channels=embed_dim, affine=learn_affine),
            'local_response': nn.LocalResponseNorm(embed_dim, alpha=eps_alpha, beta=mbval, k=kval)
        }.get(norm_name, None)
        assert self.normalizer is not None, "Unknown normalization method: {}".format(norm_name)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0, 1) which is too large!
        if learn_affine:
            self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        elif isinstance(self.normalizer, nn.LayerNorm):
            return self.normalizer(input).view(*input.size())
        else:
            return input