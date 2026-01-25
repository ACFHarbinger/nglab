"""
Normalizing Flow (RealNVP) for Time Series and Tabular Data.

This module implements a RealNVP-based Normalizing Flow model.
It maps complex data distributions to a simple base distribution (e.g., Gaussian)
through a sequence of invertible transformations.
"""

from typing import cast

import torch
from torch import nn


class CouplingLayer(nn.Module):
    """
    Affine Coupling Layer for RealNVP.

    Splits the input x into (x_a, x_b).
    x_a stays unchanged.
    x_b is transformed: y_b = x_b * exp(s(x_a)) + t(x_a)
    where s and t are scale and translation networks (MLPs).
    """

    def __init__(self, input_dim: int, hidden_dim: int, mask_type: str = "odd"):
        """Initialize Coupling Layer."""
        super().__init__()
        self.input_dim = input_dim

        # Determine split index
        # We split the input dimension in half
        self.split_idx = input_dim // 2

        # Scale and translation networks
        # Input to these networks is x_a (size: split_idx)
        # Output must be size: input_dim - split_idx
        in_channels = self.split_idx
        out_channels = input_dim - self.split_idx

        self.scale_net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
            nn.Tanh(),  # Restrict scale to avoid instability
        )

        self.translation_net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
        )

        # Mask determines which part is x_a and which is x_b
        # We handle this by slicing in forward/inverse passes instead of explicit masking tensor
        # But we need to alternate masks between layers ideally.
        # Here we rely on the user/parent class to alternate input order or just simple splitting.
        # For simple RealNVP, we can just alternate which half is transformed by flipping input before this layer?
        # Or simpler: This layer always transforms the SECOND half based on the FIRST half.
        # The parent Flow model should handle permutations/swapping.

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (Inference): x -> z
        Returns:
            z: Transformed output
            log_det_jacobian: Log determinant of the Jacobian
        """
        x_a = x[:, : self.split_idx]
        x_b = x[:, self.split_idx :]

        s = self.scale_net(x_a)
        t = self.translation_net(x_a)

        # y_a = x_a
        # y_b = x_b * exp(s) + t
        y_b = x_b * torch.exp(s) + t
        y = torch.cat([x_a, y_b], dim=1)

        # Jacobian is triangular, det is product of diagonal entries.
        # Diag entries for x_a part are 1.
        # Diag entries for x_b part are exp(s).
        # log_det = sum(log(exp(s))) = sum(s)
        log_det_jacobian = torch.sum(s, dim=1)

        return y, log_det_jacobian

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass (Generation): z -> x
        """
        z_a = z[:, : self.split_idx]
        z_b = z[:, self.split_idx :]

        s = self.scale_net(z_a)
        t = self.translation_net(z_a)

        # x_a = z_a
        # x_b = (z_b - t) * exp(-s)
        x_b = (z_b - t) * torch.exp(-s)
        x = torch.cat([z_a, x_b], dim=1)

        return x


class NormalizingFlow(nn.Module):
    """
    RealNVP Normalizing Flow Model.

    Stacks multiple CouplingLayers.
    Supports flattening of time-series inputs.
    """

    def __init__(
        self,
        input_dim: int,  # Feature dim if time-series, or total dim if flat
        num_layers: int = 4,
        hidden_dim: int = 64,
        seq_len: int = 1,  # If > 1, input is (B, L, D) and we flatten it
    ):
        """Initialize Normalizing Flow."""
        super().__init__()

        self.seq_len = seq_len
        self.feature_dim = input_dim
        self.total_dim = input_dim * seq_len

        self.layers = nn.ModuleList()

        for _i in range(num_layers):
            self.layers.append(CouplingLayer(self.total_dim, hidden_dim))
            # We add a permutation step between layers implicitly or explicitly?
            # A simple way is to define a fixed permutation.
            # Here we will just reverse the channel mapping after each coupling layer
            # to ensure all variables get transformed.

        self.base_dist = torch.distributions.Normal(0, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (Inference): x -> z, log_det
        Computes the latent representation and the log determinant of the Jacobian.
        """
        batch_size = x.shape[0]

        # Flatten if sequence
        if x.dim() > 2:
            x = x.view(batch_size, -1)

        z = x
        log_det_sum = torch.zeros(batch_size, device=x.device)

        for i, layer in enumerate(self.layers):
            z, log_det = layer(z)
            log_det_sum += log_det

            # Permute (Reverse) to mix information
            # We do this for all but the last layer usually, or all.
            # Since we want the inverse to correspond, we must enable reversing in inverse too.
            if i < len(self.layers) - 1:
                z = torch.flip(z, [1])

        return z, log_det_sum

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass (Generation): z -> x
        """
        batch_size = z.shape[0]

        for i in reversed(range(len(self.layers))):
            # Undo permutation if we did it in forward
            if i < len(self.layers) - 1:
                z = torch.flip(z, [1])

            layer = cast(CouplingLayer, self.layers[i])
            z = layer.inverse(z)

        # Reshape to sequence if needed
        if self.seq_len > 1:
            z = z.view(batch_size, self.seq_len, self.feature_dim)

        return z

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from the model.
        """
        z = torch.randn(num_samples, self.total_dim, device=device)
        return self.inverse(z)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log likelihood of input x.
        log p(x) = log p(z) + log |det J|
        """
        z, log_det = self.forward(x)

        # log p(z) (standard normal)
        # -0.5 * (z^2 + log(2pi))
        log_pz = -0.5 * (z.pow(2) + torch.log(torch.tensor(2 * torch.pi).to(x.device)))
        log_pz = torch.sum(log_pz, dim=1)

        return log_pz + log_det


def flow_loss(z: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
    """
    Compute the Negative Log Likelihood (NLL) loss.
    Minimize NLL = Maximize LL
    Loss = - (log p(z) + log_det)
    """
    # log p(z) for standard normal
    log_pz = -0.5 * (z.pow(2) + torch.log(torch.tensor(2 * torch.pi).to(z.device)))
    log_pz_sum = torch.sum(log_pz, dim=1)

    log_likelihood = log_pz_sum + log_det
    return -torch.mean(log_likelihood)
