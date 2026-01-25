"""
PyTorch Lightning Module for VAE Training

This module provides a Lightning wrapper for training the VAE model
with support for beta-VAE, KL annealing, and various reconstruction losses.
"""

from typing import Any, Literal, cast

import torch
from pytorch_lightning import LightningModule

from python.src.models.deep.autoencoders.vae import VAE, vae_loss


class VAELightningModule(LightningModule):
    """
    PyTorch Lightning module for training Variational Auto-Encoders.

    Features:
    - Beta-VAE support for controlling latent capacity
    - KL annealing for stable training
    - Multiple reconstruction loss types
    - Logging of reconstruction quality and latent statistics
    - Sample generation during validation

    Args:
        input_dim: Number of input features
        latent_dim: Dimensionality of the latent space
        d_model: Hidden dimension for backbone models
        seq_len: Input sequence length
        pred_len: Prediction/output sequence length
        encoder_type: Type of encoder backbone
        decoder_type: Type of decoder backbone
        n_layers: Number of layers
        n_heads: Number of attention heads (for Transformer)
        d_ff: Feed-forward dimension (for Transformer)
        dropout: Dropout rate
        activation: Activation function
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        kl_weight: Beta parameter for beta-VAE (1.0 = standard VAE)
        kl_anneal_epochs: Number of epochs to anneal KL weight from 0 to kl_weight
        reconstruction_loss: Type of reconstruction loss ('mse', 'l1', 'huber')
        num_val_samples: Number of samples to generate during validation
        encoder_kwargs: Additional kwargs for encoder
        decoder_kwargs: Additional kwargs for decoder
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
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        kl_weight: float = 1.0,
        kl_anneal_epochs: int = 0,
        reconstruction_loss: Literal["mse", "l1", "huber"] = "mse",
        num_val_samples: int = 4,
        encoder_kwargs: dict[str, Any] | None = None,
        decoder_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize VAELightningModule."""
        super().__init__()
        self.save_hyperparameters()

        # Initialize VAE model
        self.model = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            seq_len=seq_len,
            pred_len=pred_len,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            encoder_kwargs=encoder_kwargs,
            decoder_kwargs=decoder_kwargs,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight
        self.kl_anneal_epochs = kl_anneal_epochs
        self.reconstruction_loss = reconstruction_loss
        self.num_val_samples = num_val_samples

    def get_current_kl_weight(self) -> float:
        """
        Compute current KL weight with optional annealing.

        KL annealing linearly increases the KL weight from 0 to target value
        over the specified number of epochs. This helps prevent posterior collapse.
        """
        if self.kl_anneal_epochs == 0:
            return self.kl_weight

        current_epoch = self.current_epoch
        if current_epoch >= self.kl_anneal_epochs:
            return self.kl_weight
        else:
            return float(self.kl_weight * (current_epoch / self.kl_anneal_epochs))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through VAE."""
        return cast(dict[str, torch.Tensor], self.model(x))

    def compute_loss(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Compute VAE loss for a batch.

        Args:
            batch: Dictionary containing 'Price' tensor (batch_size, seq_len, input_dim)
            batch_idx: Batch index

        Returns:
            Dictionary with loss and metrics
        """
        # Extract input sequence
        x = batch["Price"]  # (batch_size, seq_len, input_dim)

        # Target is the future prediction window
        # For autoregressive prediction, we use the last pred_len steps
        target = x[:, -self.model.pred_len :, :]  # (batch_size, pred_len, input_dim)

        # Forward pass
        output = self.model(x)
        reconstruction = output["reconstruction"]
        mean = output["mean"]
        log_var = output["log_var"]

        # Compute loss
        current_kl_weight = self.get_current_kl_weight()
        _total_loss, loss_dict = vae_loss(
            reconstruction=reconstruction,
            target=target,
            mean=mean,
            log_var=log_var,
            kl_weight=current_kl_weight,
            reconstruction_loss=cast(Any, self.reconstruction_loss),
        )

        # Add latent statistics
        loss_dict["latent_mean"] = mean.mean()
        loss_dict["latent_std"] = torch.exp(0.5 * log_var).mean()
        loss_dict["kl_weight_current"] = torch.tensor(
            float(current_kl_weight), device=cast(torch.device, self.device)
        )

        return loss_dict

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        loss_dict = self.compute_loss(batch, batch_idx)

        # Log metrics
        self.log("train/loss", loss_dict["total_loss"], prog_bar=True)
        self.log("train/reconstruction_loss", loss_dict["reconstruction_loss"])
        self.log("train/kl_loss", loss_dict["kl_loss"])
        self.log("train/kl_weighted", loss_dict["kl_weighted"], prog_bar=True)
        self.log("train/latent_mean", loss_dict["latent_mean"])
        self.log("train/latent_std", loss_dict["latent_std"])
        self.log("train/kl_weight", loss_dict["kl_weight_current"])

        return loss_dict["total_loss"]

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        loss_dict = self.compute_loss(batch, batch_idx)

        # Log metrics
        self.log("val/loss", loss_dict["total_loss"], prog_bar=True)
        self.log("val/reconstruction_loss", loss_dict["reconstruction_loss"])
        self.log("val/kl_loss", loss_dict["kl_loss"])
        self.log("val/kl_weighted", loss_dict["kl_weighted"])
        self.log("val/latent_mean", loss_dict["latent_mean"])
        self.log("val/latent_std", loss_dict["latent_std"])

        return loss_dict["total_loss"]

    def on_validation_epoch_end(self) -> None:
        """
        Generate samples at the end of validation epoch.
        This helps visualize the quality of the learned latent space.
        """
        if self.num_val_samples > 0:
            device = cast(torch.device, self.device)
            samples = self.model.sample(num_samples=self.num_val_samples, device=device)
            # Log sample statistics
            self.log("val/sample_mean", samples.mean())
            self.log("val/sample_std", samples.std())

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        T_max = self.trainer.max_epochs if self.trainer is not None else 100
        # Ensure T_max is int
        T_max = int(T_max) if T_max is not None else 100

        # Cosine annealing with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=self.learning_rate * 0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, torch.Tensor]:
        """
        Prediction step for inference.

        Returns reconstructions and latent representations.
        """
        x = batch["Price"]

        # Get reconstruction (using mean of latent distribution for deterministic output)
        reconstruction = self.model.reconstruct(x, use_mean=True)

        # Get latent representation
        mean, log_var = self.model.encode(x)

        return {
            "reconstruction": reconstruction,
            "latent_mean": mean,
            "latent_log_var": log_var,
            "input": x,
        }
