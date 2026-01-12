"""
Non-stationary Transformer for Time Series Forecasting.
Adapted from the Time-Series-Library.
"""
import torch
import torch.nn as nn

from .modules import DataEmbedding
from models.nstransformer_networks import (
    Encoder, EncoderLayer,
    Decoder, DecoderLayer
)


# Adapted from the Time-Series-Library (https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py)
class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
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
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

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
    def __init__(self, pred_len, seq_len, input_dim, embed_dim, hidden_dim, output_dim, learner_dims, embed_type='fixed', freq='h', n_enc_layers=2, n_dec_layers=2, n_learner_layers=2, n_heads=8, dropout_rate=0.1):
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
        self.init_embedding = DataEmbedding(input_dim, embed_dim, embed_type, freq, dropout_rate)
        self.encoder = Encoder(
            [
                EncoderLayer(n_heads, embed_dim, hidden_dim, dropout_rate, normalization='layer') for _ in range(n_enc_layers)
            ],
            norm_layer=torch.nn.LayerNorm(embed_dim)
        )
        self.dec_embedding = DataEmbedding(embed_dim, embed_dim, embed_type, freq, dropout_rate)
        self.decoder = Decoder(
            [
                DecoderLayer(n_heads, embed_dim, hidden_dim, dropout_rate, normalization='layer') for _ in range(n_dec_layers)
            ],
            norm_layer=torch.nn.LayerNorm(embed_dim),
            projection=nn.Linear(embed_dim, output_dim)
        )
        self.tau_learner = Projector(input_dim, seq_len, learner_dims, n_learner_layers, output_dim=1)
        self.delta_learner = Projector(input_dim, seq_len, learner_dims, n_learner_layers, output_dim=seq_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting function.
        """
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc).exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)

        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                            dim=1).to(x_enc.device).clone()

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, tau=tau, delta=delta)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass for the NSTransformer.
        """
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
