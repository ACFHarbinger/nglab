"""
Variational Auto-Encoder (VAE) for Time Series Prediction

This module implements a VAE architecture specifically designed for time series data,
with support for multiple backbone architectures (Transformer, Mamba, LSTM, etc.)
"""

from typing import Any, Literal, cast

import torch
from torch import nn


class VAE(nn.Module):
    """
    Variational Auto-Encoder for time series prediction.

    Architecture:
    - Encoder: TimeSeriesBackbone (Transformer/Mamba/LSTM/etc.) → latent mean & log_variance
    - Reparameterization: Sample from N(mean, variance) using the reparameterization trick
    - Decoder: Latent vector → TimeSeriesBackbone → reconstructed sequence
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        latent_dim: int,
        d_model: int = 128,
        seq_len: int = 100,
        pred_len: int = 20,
        encoder_type: Literal["transformer", "mamba", "lstm", "gru", "xlstm"] = "mamba",
        decoder_type: (
            Literal["transformer", "mamba", "lstm", "gru", "xlstm"] | None
        ) = None,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        encoder_kwargs: dict[str, Any] | None = None,
        decoder_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the VAE.

        See class docstring for arguments details.
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type or encoder_type

        # Map types to Backbone names
        def _map_name(name: str) -> str:
            n = name.lower()
            if n == "lstm":
                return "LSTM"
            if n == "gru":
                return "GRU"
            if n == "mamba":
                return "Mamba"
            if n == "xlstm":
                return "xLSTM"
            if n == "transformer":
                return "NSTransformer"
            return name

        backbone_encoder_type = _map_name(encoder_type)
        backbone_decoder_type = _map_name(self.decoder_type)

        from python.src.models.time_series import TimeSeriesBackbone

        # Encoder: maps input sequence to embedding space
        encoder_cfg: dict[str, Any] = {
            "name": backbone_encoder_type,
            "feature_dim": input_dim,
            "hidden_dim": d_model,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "num_layers": n_layers,
            "dropout": dropout,
            "activation": activation,
            "num_heads": n_heads,
            "d_ff": d_ff,
            "embed_dim": d_model,
            "output_dim": d_model,
            **(encoder_kwargs or {}),
        }

        self.encoder = TimeSeriesBackbone(encoder_cfg)

        # Project embeddings to latent distribution parameters
        self.fc_mean = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)

        # Project latent sample to decoder initial state
        self.latent_to_decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )

        # Expand latent to sequence for decoder
        self.latent_expander = nn.Linear(d_model, d_model * pred_len)

        # Decoder: maps latent representation back to sequence space
        decoder_cfg: dict[str, Any] = {
            "name": backbone_decoder_type,
            "feature_dim": d_model,
            "output_dim": input_dim,
            "hidden_dim": d_model,
            "seq_len": pred_len,
            "pred_len": pred_len,
            "num_layers": n_layers,
            "dropout": dropout,
            "activation": activation,
            "num_heads": n_heads,
            "d_ff": d_ff,
            "embed_dim": d_model,
            "return_sequence": True,
            "output_type": "prediction",
            **(decoder_kwargs or {}),
        }

        self.decoder = TimeSeriesBackbone(decoder_cfg)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            mean: Latent mean of shape (batch_size, latent_dim)
            log_var: Latent log variance of shape (batch_size, latent_dim)
        """
        # Get embedding from encoder backbone
        embedding = self.encoder(x)  # (batch_size, d_model)

        # Project to latent distribution parameters
        mean = self.fc_mean(embedding)
        log_var = self.fc_log_var(embedding)

        return mean, log_var

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mean + std * epsilon
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output sequence.

        Args:
            z: Latent vector of shape (batch_size, latent_dim)

        Returns:
            reconstruction: Reconstructed sequence of shape (batch_size, pred_len, input_dim)
        """
        # Project latent to decoder input space
        h = self.latent_to_decoder(z)  # (batch_size, d_model)

        # Expand to sequence
        h = self.latent_expander(h)  # (batch_size, d_model * pred_len)
        h = h.view(-1, self.pred_len, self.d_model)  # (batch_size, pred_len, d_model)

        # Decode to output sequence
        # Explicitly cast to Tensor to resolve no-any-return
        reconstruction = cast(torch.Tensor, self.decoder(h))

        return reconstruction

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through the VAE.
        """
        # Encode
        mean, log_var = self.encode(x)

        # Sample from latent distribution
        z = self.reparameterize(mean, log_var)

        # Decode
        reconstruction = self.decode(z)

        return {
            "reconstruction": reconstruction,
            "mean": mean,
            "log_var": log_var,
            "z": z,
        }

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples from the learned latent distribution.
        """
        # Sample from standard normal
        z = torch.randn(num_samples, self.latent_dim, device=device)

        # Decode
        with torch.no_grad():
            samples = self.decode(z)

        return samples

    def reconstruct(self, x: torch.Tensor, use_mean: bool = False) -> torch.Tensor:
        """
        Reconstruct input sequence (for inference).
        """
        mean, log_var = self.encode(x)

        if use_mean:
            z = mean
        else:
            z = self.reparameterize(mean, log_var)

        return self.decode(z)


def vae_loss(  # noqa: PLR0913
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1.0,
    reconstruction_loss: Literal["mse", "l1", "huber"] = "mse",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute VAE loss = Reconstruction Loss + KL Divergence.
    """
    # Reconstruction loss
    if reconstruction_loss == "mse":
        recon_loss = nn.functional.mse_loss(reconstruction, target, reduction="mean")
    elif reconstruction_loss == "l1":
        recon_loss = nn.functional.l1_loss(reconstruction, target, reduction="mean")
    elif reconstruction_loss == "huber":
        recon_loss = nn.functional.huber_loss(reconstruction, target, reduction="mean")
    else:
        raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    kl_loss = kl_loss / mean.size(0)  # Normalize by batch size

    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss

    loss_dict = {
        "total_loss": total_loss,
        "reconstruction_loss": recon_loss,
        "kl_loss": kl_loss,
        "kl_weighted": kl_weight * kl_loss,
    }

    return total_loss, loss_dict
