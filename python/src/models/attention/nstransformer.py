"""
Non-stationary Transformer for Time Series Forecasting.
Adapted from the Time-Series-Library.
"""

from typing import Any

import torch
from torch import nn

from python.src.models.modules import (
    AttentionLayer,
    DataEmbedding,
    DSAttention,
    Normalization,
    SkipConnection,
    Transpose,
)


class EncoderLayer(nn.Module):
    """
    Encoder layer for Non-Stationary Transformer.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        hidden_dim: int,
        dropout_rate: float = 0.1,
        normalization: str = "layer",
    ) -> None:
        """
        Initialize.
        """
        super().__init__()
        self.attention = SkipConnection(
            AttentionLayer(DSAttention(False, dropout_rate, False), embed_dim, n_heads)
        )
        self.conv = nn.Sequential(
            Transpose(),
            nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(in_channels=hidden_dim, out_channels=embed_dim, kernel_size=1),
            Transpose(),
            nn.Dropout(dropout_rate),
        )
        self.norm = Normalization(embed_dim, normalization)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Any,
        tau: torch.Tensor | None,
        delta: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Forward pass.
        """
        y = x = torch.as_tensor(
            self.norm(
                self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)[0]
            )
        )
        y = torch.as_tensor(self.conv(y))
        return torch.as_tensor(self.norm(x + y))


class Encoder(nn.Module):
    """
    Encoder network consisting of multiple layers.
    """

    def __init__(
        self,
        attn_layers: list[EncoderLayer],
        conv_layers: list[nn.Module] | None = None,
        norm_layer: nn.Module | None = None,
    ) -> None:
        """
        Initialize.
        """
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Any = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        """
        # x [B, L, D]
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers, strict=False)
            ):
                delta_val = delta if i == 0 else None
                out = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta_val)
                if isinstance(out, tuple | list):
                    x = out[0]
                else:
                    x = out
                x = torch.as_tensor(conv_layer(x))

            # Final attention layer
            last_out = self.attn_layers[-1](x, attn_mask=attn_mask, tau=tau, delta=None)
            if isinstance(last_out, tuple | list):
                x = last_out[0]
            else:
                x = last_out
        else:
            for attn_layer in self.attn_layers:
                out_loop = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                if isinstance(out_loop, tuple | list):
                    x = out_loop[0]
                else:
                    x = out_loop

        if self.norm is not None:
            x = torch.as_tensor(self.norm(x))

        return x


class DecoderLayer(nn.Module):
    """
    Decoder layer for Non-Stationary Transformer.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        hidden_dim: int,
        dropout_rate: float = 0.1,
        normalization: str = "layer",
    ) -> None:
        """
        Initialize.
        """
        super().__init__()
        self.attention = SkipConnection(
            AttentionLayer(DSAttention(True, dropout_rate, False), embed_dim, n_heads)
        )
        self.cross_attention = SkipConnection(
            AttentionLayer(DSAttention(False, dropout_rate, False), embed_dim, n_heads)
        )
        self.conv = nn.Sequential(
            Transpose(),
            nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(in_channels=hidden_dim, out_channels=embed_dim, kernel_size=1),
            Transpose(),
            nn.Dropout(dropout_rate),
        )
        self.norm = Normalization(embed_dim, normalization)

    def forward(  # noqa: PLR0913
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: Any = None,
        cross_mask: Any = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        """
        x = torch.as_tensor(
            self.norm(self.attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        )
        x = torch.as_tensor(
            self.norm(
                self.cross_attention(
                    x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
                )[0]
            )
        )
        y = torch.as_tensor(self.conv(x))
        return torch.as_tensor(self.norm(x + y))


class Decoder(nn.Module):
    """
    Decoder network consisting of multiple layers.
    """

    def __init__(
        self,
        layers: list[DecoderLayer],
        norm_layer: nn.Module | None = None,
        projection: nn.Module | None = None,
    ) -> None:
        """
        Initialize.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(  # noqa: PLR0913
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: Any = None,
        cross_mask: Any = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        """
        for layer in self.layers:
            x = torch.as_tensor(
                layer(
                    x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
                )
            )

        if self.norm is not None:
            x = torch.as_tensor(self.norm(x))

        if self.projection is not None:
            x = torch.as_tensor(self.projection(x))
        return x


class Projector(nn.Module):
    """
    MLP to learn the De-stationary factors
    """

    def __init__(  # noqa: PLR0913
        self,
        enc_in: int,
        seq_len: int,
        hidden_dims: list[int],
        hidden_layers: int,
        output_dim: int,
        kernel_size: int = 3,
    ) -> None:
        """
        Initialize the Projector.
        """
        super().__init__()

        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.series_conv = nn.Conv1d(
            in_channels=seq_len,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Projector.
        """
        batch_size = x.shape[0]
        x = torch.as_tensor(self.series_conv(x))  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = torch.as_tensor(self.backbone(x))  # B x O

        return y

from python.src.utils.registry import register_model


@register_model("nstransformer")
class NSTransformer(nn.Module):
    """
    Non-stationary Transformer Model.
    """

    def __init__(  # noqa: PLR0913
        self,
        pred_len: int,
        seq_len: int,
        input_dim: int,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        learner_dims: list[int],
        embed_type: str = "fixed",
        freq: str = "h",
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        n_learner_layers: int = 2,
        n_heads: int = 8,
        dropout_rate: float = 0.1,
    ) -> None:
        """
        Initialize the NSTransformer.
        """
        super().__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.init_embedding = DataEmbedding(
            input_dim, embed_dim, embed_type, freq, dropout_rate
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    n_heads, embed_dim, hidden_dim, dropout_rate, normalization="layer"
                )
                for _ in range(n_enc_layers)
            ],
            norm_layer=torch.nn.LayerNorm(embed_dim),
        )
        self.dec_embedding = DataEmbedding(
            embed_dim, embed_dim, embed_type, freq, dropout_rate
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    n_heads, embed_dim, hidden_dim, dropout_rate, normalization="layer"
                )
                for _ in range(n_dec_layers)
            ],
            norm_layer=torch.nn.LayerNorm(embed_dim),
            projection=nn.Linear(embed_dim, output_dim),
        )
        self.tau_learner = Projector(
            input_dim, seq_len, learner_dims, n_learner_layers, output_dim=1
        )
        self.delta_learner = Projector(
            input_dim, seq_len, learner_dims, n_learner_layers, output_dim=seq_len
        )
        self.label_len: int = 0

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Forecasting function.
        """
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        tau = torch.as_tensor(self.tau_learner(x_raw, std_enc)).exp()
        delta = torch.as_tensor(self.delta_learner(x_raw, mean_enc))

        x_dec_new = (
            torch.cat(
                [
                    x_enc[:, -self.label_len :, :],
                    torch.zeros_like(x_dec[:, -self.pred_len :, :]),
                ],
                dim=1,
            )
            .to(x_enc.device)
            .clone()
        )

        enc_out = torch.as_tensor(self.init_embedding(x_enc, x_mark_enc))
        enc_out = torch.as_tensor(
            self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)
        )

        dec_out = torch.as_tensor(self.dec_embedding(x_dec_new, x_mark_dec))
        dec_out = torch.as_tensor(
            self.decoder(
                dec_out, enc_out, x_mask=None, cross_mask=None, tau=tau, delta=delta
            )
        )
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor | None,
        mask: Any = None,
    ) -> torch.Tensor:
        """
        Forward pass for the NSTransformer.
        """
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
