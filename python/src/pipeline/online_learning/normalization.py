"""Online normalization utilities."""

import torch
from torch import nn


class OnlineNormalizer(nn.Module):
    """
    Online normalization layer using Welford's algorithm for running mean and variance.
    Can be used as a drop-in layer in PyTorch models.
    """

    def __init__(
        self,
        num_features: int,
        momentum: float | None = None,
        affine: bool = True,
        eps: float = 1e-5,
    ) -> None:
        """
        Args:
            num_features: Number of features to normalize.
            momentum: Momentum for moving average. If None, uses exact cumulative stats (Welford).
                      If float (e.g., 0.1), uses exponential moving average.
            affine: Whether to learn learnable affine parameters (weight/bias) like BatchNorm.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.affine = affine
        self.eps = eps

        # Buffer for running stats (not trained via backprop)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("count", torch.tensor(0.0))

        self.running_mean: torch.Tensor
        self.running_var: torch.Tensor
        self.count: torch.Tensor

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset(self) -> None:
        """Reset running statistics."""
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.count.zero_()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def update(self, x: torch.Tensor) -> None:
        """
        Update running statistics without applying normalization.
        Expects x of shape (batch_size, num_features) or (num_features,).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.size(0)

        if self.momentum is None:
            # Welford's algorithm / Combined Variance
            delta = batch_mean - self.running_mean
            total_count = self.count + batch_count

            # Update mean
            new_mean = self.running_mean + delta * (batch_count / total_count)

            # Update sum of squares (m2 = var * count)
            m2_old = self.running_var * self.count
            m2_new = batch_var * batch_count
            m2_total = (
                m2_old + m2_new + delta**2 * self.count * batch_count / total_count
            )

            self.running_mean = new_mean
            self.running_var = m2_total / total_count
            self.count = total_count
        else:
            # Exponential Moving Average (Momentum)
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var
            # Count is irrelevant for momentum

    def forward(self, x: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """
        Forward pass: Normalize x using running stats.
        If training or update_stats=True, updates the running stats with the current batch.
        """
        if self.training and update_stats:
            with torch.no_grad():
                self.update(x)

        # Normalize
        norm_x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        if self.affine:
            norm_x = norm_x * self.weight + self.bias

        return norm_x
