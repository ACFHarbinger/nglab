"""Normalization layer wrapper supporting various types (Batch, Layer, Instance)."""

import math
from typing import cast

import torch
from torch import nn


class Normalization(nn.Module):
    """
    Wrapper for various Normalization layers (Batch, Layer, Instance).
    """

    def __init__(  # noqa: PLR0913
        self,
        embed_dim: int,
        norm_name: str = "batch",
        eps_alpha: float = 1e-05,
        learn_affine: bool = True,
        track_stats: bool = False,
        mbval: float = 0.1,
        n_groups: int | None = None,
        kval: float = 1.0,
        bias: bool = True,
    ) -> None:
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
        super().__init__()

        self.normalizer: nn.Module

        if norm_name == "instance":
            self.normalizer = nn.InstanceNorm1d(
                embed_dim,
                eps=eps_alpha,
                affine=learn_affine,
                track_running_stats=track_stats,
                momentum=mbval,
            )
        elif norm_name == "batch":
            self.normalizer = nn.BatchNorm1d(
                embed_dim,
                eps=eps_alpha,
                affine=learn_affine,
                track_running_stats=track_stats,
                momentum=mbval,
            )
        elif norm_name == "layer":
            self.normalizer = nn.LayerNorm(
                embed_dim, eps=eps_alpha, elementwise_affine=learn_affine, bias=bias
            )
        elif norm_name == "group":
            actual_n_groups = n_groups if n_groups is not None else 1
            self.normalizer = nn.GroupNorm(
                actual_n_groups,
                eps=eps_alpha,
                num_channels=embed_dim,
                affine=learn_affine,
            )
        elif norm_name == "local_response":
            self.normalizer = nn.LocalResponseNorm(
                embed_dim, alpha=eps_alpha, beta=mbval, k=kval
            )
        else:
            raise ValueError(f"Unknown normalization method: {norm_name}")

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0, 1) which is too large!
        if learn_affine:
            self.init_parameters()

    def init_parameters(self) -> None:
        """Initializes the affine parameters if applicable."""
        for param in self.parameters():
            if param.dim() > 0:
                stdv = 1.0 / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Applies the normalization to the input.

        Args:
            x: Input tensor.
            mask: Optional mask (currently not used by normalization layers).

        Returns:
            Normalized tensor.
        """
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return cast(torch.Tensor, self.normalizer(x.view(-1, x.size(-1)))).view(
                *x.size()
            )
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            # InstanceNorm1d expects (N, C, L)
            return cast(torch.Tensor, self.normalizer(x.permute(0, 2, 1))).permute(
                0, 2, 1
            )
        elif isinstance(self.normalizer, nn.LayerNorm):
            return cast(torch.Tensor, self.normalizer(x)).view(*x.size())
        elif isinstance(self.normalizer, nn.GroupNorm | nn.LocalResponseNorm):
            # GroupNorm usually expects (N, C, ...)
            if x.dim() == 3:
                return cast(torch.Tensor, self.normalizer(x.transpose(1, 2))).transpose(
                    1, 2
                )
            return cast(torch.Tensor, self.normalizer(x))
        else:
            return x
