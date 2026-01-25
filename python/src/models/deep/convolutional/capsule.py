"""
Capsule Network Layer.
"""

import torch
from torch import nn


class CapsuleLayer(nn.Module):
    """
    Simplified Capsule Layer (CN).
    """

    def __init__(
        self,
        in_caps: int,
        in_dim: int,
        out_caps: int,
        out_dim: int,
        output_type: str = "embedding",
    ) -> None:
        """
        Initialize Capsule Layer.
        """
        super().__init__()
        self.in_caps = in_caps
        self.in_dim = in_dim
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.output_type = output_type

        self.W = nn.Parameter(torch.randn(out_caps, in_caps, out_dim, in_dim) * 0.1)

    def squash(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Squash activation function.
        Scales vector length to [0, 1].
        """
        norm_sq = torch.sum(x**2, dim=dim, keepdim=True)
        scale = norm_sq / (1 + norm_sq) / (torch.sqrt(norm_sq) + 1e-8)
        return scale * x

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        x: (Batch, Seq, In_Caps, In_Dim) or (Batch, In_Caps, In_Dim)
        """
        if x.dim() == 4:
            b, s, c, d = x.shape
            x_flat = x.view(b * s, c, d)
            v_flat = self._process(x_flat)
            res = v_flat.view(b, s, self.out_caps, self.out_dim)
        else:
            res = self._process(x)

        if not return_sequence and res.dim() == 4:
            return res[:, -1, :, :]
        return res

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, In_Caps, In_Dim)
        x = x.unsqueeze(1).unsqueeze(4)
        u_hat = torch.matmul(self.W, x)
        u_hat = u_hat.squeeze(4)
        v = torch.sum(u_hat, dim=2)
        return self.squash(v)
