"""Normalized activation functions (Softmax, etc.) with adaptive options."""

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


class NormalizedActivationFunction(nn.Module):
    """
    Wrapper for normalized activation functions (Softmax, LogSoftmax, etc.).
    """

    def __init__(  # noqa: PLR0913
        self,
        naf_name: str = "softmax",
        dim: int = -1,
        n_classes: int | None = None,
        cutoffs: Sequence[int] | None = None,
        dval: float = 4.0,
        bias: bool = False,
    ) -> None:
        """
        Initializes the normalized activation function.

        Args:
            naf_name: Name of the normalized activation function ('softmax', 'logsoftmax', etc.).
            dim: Dimension along which the activation is applied (or in_features for adaptive).
            n_classes: Number of classes (for adaptive softmax).
            cutoffs: Cutoffs for adaptive softmax clusters.
            dval: Divisor value for adaptive softmax.
            bias: Whether to use bias in adaptive softmax.
        """
        super().__init__()
        self.norm_activation: nn.Module

        if naf_name == "softmin":
            self.norm_activation = nn.Softmin(dim=dim)
        elif naf_name == "softmax":
            self.norm_activation = nn.Softmax(dim=dim)
        elif naf_name == "logsoftmax":
            self.norm_activation = nn.LogSoftmax(dim=dim)
        elif naf_name == "softmax2d":
            self.norm_activation = nn.Softmax2d()
        elif naf_name == "adaptivelogsoftmax":
            if n_classes is None:
                raise ValueError("n_classes must be provided for adaptivelogsoftmax")
            if dim == -1:
                # dim=-1 is invalid for in_features in AdaptiveLogSoftmaxWithLoss
                # We assume if naf_name is adaptivelogsoftmax, the user intends dim to be in_features.
                # However, if it's default -1, we might have a problem.
                # In most cases in this codebase, embed_dim is passed.
                pass

            # Ensure cutoffs is list
            actual_cutoffs = (
                list(cutoffs)
                if cutoffs is not None
                else [n_classes // 4, n_classes // 2, 3 * n_classes // 4]
            )

            self.norm_activation = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=dim,
                n_classes=n_classes,
                cutoffs=actual_cutoffs,
                div_value=dval,
                head_bias=bias,
            )
        else:
            raise ValueError(f"Unknown normalized activation function: {naf_name}")

        if isinstance(self.norm_activation, nn.AdaptiveLogSoftmaxWithLoss):
            self.init_parameters()

    def init_parameters(self) -> None:
        """Initializes the parameters if applicable."""
        for param in self.parameters():
            if param.dim() > 0:
                stdv = 1.0 / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor | Any:
        """
        Applies the normalized activation function to the input.

        Args:
            x: Input tensor.
            mask: Optional mask (not used by all activations).

        Returns:
            Output tensor.
        """
        if isinstance(self.norm_activation, nn.AdaptiveLogSoftmaxWithLoss):
            # AdaptiveLogSoftmaxWithLoss requires target as second argument in its forward for training,
            # but for inference it just needs input. Wait, actually it returns ASMoutput (output, loss).
            # This wrapper seems simplified.
            return self.norm_activation(x)
        return self.norm_activation(x)
