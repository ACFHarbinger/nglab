"""
Non-stationary Transformer for Time Series Forecasting.
Adapted from the Time-Series-Library.
"""

import torch
from torch import nn

from python.src.models.deep.modules import (
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
        self, n_heads, embed_dim, hidden_dim, dropout_rate=0.1, normalization="layer"
    ):
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

    def forward(self, x, attn_mask, tau, delta):
        """
        Forward pass.
        """
        y = x = self.norm(
            self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        )
        y = self.conv(y)
        return self.norm(x + y)


class Encoder(nn.Module):
    """
    Encoder network consisting of multiple layers.
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        """
        Initialize.
        """
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        Forward pass.
        """
        # x [B, L, D]
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers, strict=False)
            ):
                delta = delta if i == 0 else None
                x = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
            x = self.attn_layers[-1](x, tau=tau, delta=None)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        return x


class DecoderLayer(nn.Module):
    """
    Decoder layer for Non-Stationary Transformer.
    """

    def __init__(
        self, n_heads, embed_dim, hidden_dim, dropout_rate=0.1, normalization="layer"
    ):
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
        self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None
    ):
        """
        Forward pass.
        """
        x = self.norm(self.attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm(
            self.cross_attention(
                x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
            )[0]
        )
        y = self.conv(x)
        return self.norm(x + y)


class Decoder(nn.Module):
    """
    Decoder network consisting of multiple layers.
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        """
        Initialize.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(  # noqa: PLR0913
        self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None
    ):
        """
        Forward pass.
        """
        for layer in self.layers:
            x = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


# Adapted from the Time-Series-Library (https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py)
class Projector(nn.Module):
    """
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(  # noqa: PLR0913
        self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3
    ):
        """
        Initialize the Projector.

        Args:
            enc_in (int): Number of input channels.
            seq_len (int): Input sequence length.
            hidden_dims (list): List of hidden layer dimensions.
            hidden_layers (int): Number of hidden layers.
            output_dim (int): Output dimension.
            kernel_size (int): Convolution kernel size.
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

    def forward(self, x, stats):
        """
        Forward pass for the Projector.

        Args:
            x (Tensor): Input sequence.
            stats (Tensor): Statistics (e.g., mean/std).

        Returns:
            Tensor: Projection output.
        """
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


# Based on https://github.com/thuml/Nonstationary_Transformers
class NSTransformer(nn.Module):
    """
    Non-stationary Transformer Model.
    """

    def __init__(  # noqa: PLR0913
        self,
        pred_len,
        seq_len,
        input_dim,
        embed_dim,
        hidden_dim,
        output_dim,
        learner_dims,
        embed_type="fixed",
        freq="h",
        n_enc_layers=2,
        n_dec_layers=2,
        n_learner_layers=2,
        n_heads=8,
        dropout_rate=0.1,
    ):
        """
        Initialize the NSTransformer.

        Args:
            pred_len (int): Prediction sequence length.
            seq_len (int): Input sequence length.
            input_dim (int): Input dimension.
            embed_dim (int): Embedding dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
            learner_dims (list): Dimensions for the stationary learner.
            embed_type (str): Type of embedding.
            freq (str): Frequency of time features.
            n_enc_layers (int): Number of encoder layers.
            n_dec_layers (int): Number of decoder layers.
            n_learner_layers (int): Number of learner layers.
            n_heads (int): Number of attention heads.
            dropout_rate (float): Dropout rate.
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

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
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
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc).exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)

        if not hasattr(self, "label_len"):
            self.label_len = 0  # Default to 0 if not provided

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

        enc_out = self.init_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, tau=tau, delta=delta
        )
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass for the NSTransformer.
        """
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
