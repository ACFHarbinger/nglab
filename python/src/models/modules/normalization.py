"""Normalization layer wrapper supporting various types (Batch, Layer, Instance)."""
import math
import torch.nn as nn

from typing import Optional


class Normalization(nn.Module):
    """
    Wrapper for various Normalization layers (Batch, Layer, Instance).
    """
    def __init__(self, 
                embed_dim: int, 
                norm_name: str='batch', 
                eps_alpha: float=1e-05, 
                learn_affine: Optional[bool]=True, 
                track_stats: Optional[bool]=False,  
                mbval: Optional[float]=0.1, 
                n_groups: Optional[int]=None, 
                kval: Optional[float]=None,
                bias: Optional[bool]=True):
        """
        Initializes the normalization layer.

        Args:
            embed_dim: Embedding dimension.
            norm_name: Type of normalization ('batch', 'layer', 'instance', 'group', 'local_response').
            eps_alpha: Epsilon value for numerical stability.
            learn_affine: If True, learn affine parameters (weight and bias).
            track_stats: If True, track running statistics for BatchNorm/InstanceNorm.
            mbval: Momentum for running statistics or beta for LocalResponseNorm.
            n_groups: Number of groups for GroupNorm.
            kval: k value for LocalResponseNorm.
            bias: If True, add bias for LayerNorm.
        """
        super(Normalization, self).__init__()
        
        if norm_name == 'instance':
            self.normalizer = nn.InstanceNorm1d(embed_dim, eps=eps_alpha, affine=learn_affine, track_running_stats=track_stats, momentum=mbval)
        elif norm_name == 'batch':
            self.normalizer = nn.BatchNorm1d(embed_dim, eps=eps_alpha, affine=learn_affine, track_running_stats=track_stats, momentum=mbval)
        elif norm_name == 'layer':
            self.normalizer = nn.LayerNorm(embed_dim, eps=eps_alpha, elementwise_affine=learn_affine, bias=bias)
        elif norm_name == 'group':
            if n_groups is None:
                # Fallback or error if n_groups not provided
                n_groups = 1 # avoid error if not used, or raise?
            self.normalizer = nn.GroupNorm(n_groups, eps=eps_alpha, num_channels=embed_dim, affine=learn_affine)
        elif norm_name == 'local_response':
            self.normalizer = nn.LocalResponseNorm(embed_dim, alpha=eps_alpha, beta=mbval, k=kval)
        else:
            raise ValueError(f"Unknown normalization method: {norm_name}")

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0, 1) which is too large!
        if learn_affine:
            self.init_parameters()

    def init_parameters(self):
        """Initializes the affine parameters if applicable."""
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        """
        Applies the normalization to the input.

        Args:
            input: Input tensor.
            mask: Optional mask (currently not used by normalization layers).

        Returns:
            Normalized tensor.
        """
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        elif isinstance(self.normalizer, nn.LayerNorm):
            return self.normalizer(input).view(*input.size())
        else:
            return input