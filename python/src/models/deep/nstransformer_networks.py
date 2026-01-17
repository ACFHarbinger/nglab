"""
Internal building blocks for Non-Stationary Transformer (Encoder/Decoder).
"""

from torch import nn

from ..modules import (
    AttentionLayer,
    DSAttention,
    Normalization,
    SkipConnection,
    Transpose,
)


# Adapted from the Time-Series-Library (https://github.com/thuml/Time-Series-Library/blob/main/layers/Transformer_EncDec.py)
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
        super(EncoderLayer, self).__init__()
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
        super(Encoder, self).__init__()
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
        super(DecoderLayer, self).__init__()
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

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
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
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
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
