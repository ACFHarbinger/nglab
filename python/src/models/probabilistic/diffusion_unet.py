"""
1D U-Net Model for Time Series Diffusion.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embeddings for time steps t.
    """

    def __init__(self, dim: int) -> None:
        """Initialize Sinusoidal Embeddings."""
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Generate embeddings for time steps."""
        device = time.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb_factors = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        embeddings = time[:, None].float() * emb_factors[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with optional time embedding injection and group norm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int | None = None,
        n_groups: int = 8,
    ) -> None:
        """Initialize Residual Block."""
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj: nn.Linear | None
        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, out_channels)
        else:
            self.time_proj = None

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut: nn.Module = nn.Conv1d(
                in_channels, out_channels, kernel_size=1
            )
        else:
            self.shortcut = nn.Identity()

    def forward(
        self, x: torch.Tensor, time_emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass."""
        h = self.conv1(F.silu(self.norm1(x)))

        if self.time_proj is not None and time_emb is not None:
            t = self.time_proj(F.silu(time_emb))
            h = h + t[:, :, None]  # Broadcast over time dimension

        h = self.conv2(F.silu(self.norm2(h)))
        return torch.as_tensor(h + self.shortcut(x))


class DiffusionUNet1D(nn.Module):
    """
    1D U-Net for Denoising Time Series.
    Input:
        x: (Batch, Seq_Len, Features) [We transpose to (B, C, L) internally]
        t: (Batch) Time steps
        cond: (Batch, Seq_Len, Features) Condition (e.g., historical window)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        layers: list[int] | None = None,
        time_emb_dim: int = 128,
    ) -> None:
        """Initialize Diffusion UNet."""
        if layers is None:
            layers = [1, 2, 4]
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([])
        channels = hidden_dim
        for scale in layers:
            out_channels = hidden_dim * scale
            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualBlock1D(channels, out_channels, time_emb_dim),
                        ResidualBlock1D(out_channels, out_channels, time_emb_dim),
                        nn.Conv1d(out_channels, out_channels, 3, 2, 1),  # Downsample
                    ]
                )
            )
            channels = out_channels

        # Middle
        self.mid_block1 = ResidualBlock1D(channels, channels, time_emb_dim)
        self.mid_block2 = ResidualBlock1D(channels, channels, time_emb_dim)

        # Upsample
        self.ups = nn.ModuleList([])
        for scale in reversed(layers):
            out_channels = hidden_dim * scale
            # Skip connections double the input channels
            self.ups.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose1d(channels, out_channels, 4, 2, 1),  # Upsample
                        ResidualBlock1D(
                            out_channels * 2, out_channels, time_emb_dim
                        ),  # *2 for skip concat
                        ResidualBlock1D(out_channels, out_channels, time_emb_dim),
                    ]
                )
            )
            channels = out_channels

        self.out_conv = nn.Conv1d(channels, output_dim, kernel_size=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (B, L, F)
            t: (B)
            cond: (B, L, F_cond) optional condition
        """
        if cond is not None:
            # Concat along feature dimension
            x = torch.cat([x, cond], dim=-1)

        x = x.transpose(1, 2)  # (B, C, L)

        # Time Embedding
        t_emb = self.time_mlp(t)

        # Initial
        h = self.init_conv(x)

        # Down
        skips: list[torch.Tensor] = []
        for down_layer in self.downs:
            if not isinstance(down_layer, nn.ModuleList):
                continue
            block1, block2, downsample = down_layer
            h = block1(h, t_emb)
            h = block2(h, t_emb)
            skips.append(h)
            h = downsample(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # Up
        for up_layer in self.ups:
            if not isinstance(up_layer, nn.ModuleList):
                continue
            upsample, block1, block2 = up_layer
            h = upsample(h)
            skip = skips.pop()

            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(
                    h, size=skip.shape[-1], mode="linear", align_corners=False
                )

            h = torch.cat([h, skip], dim=1)
            h = block1(h, t_emb)
            h = block2(h, t_emb)

        h = self.out_conv(h)
        return torch.as_tensor(h.transpose(1, 2))  # Back to (B, L, OUT)
