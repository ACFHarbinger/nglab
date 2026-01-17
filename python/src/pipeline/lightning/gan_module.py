"""
Generative Adversarial Network (GAN) Module for Time Series Prediction.
"""

from typing import Any

import torch
from torch import nn

from .base import BaseModule


class GANLightningModule(BaseModule):
    """
    Lightning Module for Time Series GAN (e.g., TimeGAN style or conditional forecasting GAN).

    Structure:
    - Generator: Takes history (X) -> Predicts future (Y_hat).
    - Discriminator: Takes full sequence (X + Y) -> Predicts Real/Fake.
    """

    def __init__(
        self, generator: nn.Module, discriminator: nn.Module, cfg: dict[str, Any]
    ):
        """
        Initialize the GAN module.

        Args:
            generator (nn.Module): The generator network.
            discriminator (nn.Module): The discriminator network.
            cfg (Dict[str, Any]): Configuration dictionary.
        """
        super().__init__(cfg)
        self.save_hyperparameters(ignore=["generator", "discriminator"])
        self.generator = generator
        self.discriminator = discriminator

        # Loss function
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()  # Optional auxiliary loss for reconstruction

        # Hyperparameters
        self.lambda_adv = cfg.get("lambda_adv", 1.0)
        self.lambda_l1 = cfg.get(
            "lambda_l1", 100.0
        )  # Reconstruction weight often high in cGAN (Pix2Pix style)
        self.lr_g = cfg.get("lr_g", 2e-4)  # TTUR: D often higher or balanced
        self.lr_d = cfg.get("lr_d", 2e-4)
        self.b1 = cfg.get("beta1", 0.5)
        self.b2 = cfg.get("beta2", 0.999)

        # Buffer for input/target concatenation if needed
        self.automatic_optimization = False  # We handle G and D steps manually

    def forward(self, x):
        """
        Forward pass (Generator prediction).
        """
        return self.generator(x)

    def configure_optimizers(self):
        """
        Configure optimizers for Generator and Discriminator.
        """
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr_g, betas=(self.b1, self.b2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr_d, betas=(self.b1, self.b2)
        )
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx):
        """
        GAN Training Step.
        """
        # Batch: dictionary with 'observation' (history) and 'target' (future)
        # Or implicitly handled if batch is a tuple.
        # Assuming standard NGLab dict format:
        if isinstance(batch, dict):
            x = batch.get("observation")
            y = batch.get("target")
        else:
            # Tuple assumption
            x, y = batch

        # Optimizers
        opt_g, opt_d = self.optimizers()

        # --- Train Generator ---
        # Generate fake future
        y_hat = self.generator(x)

        # Create full fake sequence: Concat history + prediction
        # Shape: (Batch, Seq_Len_X + Seq_Len_Y, Features)
        # Only if D expects full sequence.
        # If D expects just Y, modify logic.
        # Assuming D takes full joint distribution P(X, Y).

        # Check dimensions
        # x: (B, Lx, F), y: (B, Ly, F), y_hat: (B, Ly, F)
        # full_fake = torch.cat([x, y_hat], dim=1)

        # NOTE: If Feature dims don't match, we assume they do for TS prediction tasks.
        full_fake = torch.cat([x, y_hat], dim=1)

        # Adversarial ground truth
        valid = torch.ones(x.size(0), 1, device=self.device)
        fake = torch.zeros(x.size(0), 1, device=self.device)

        # 1. Generator Update
        self.toggle_optimizer(opt_g)

        # Discriminator pass on fake
        # D output: (Batch, 1) or (Batch, Seq, 1)? Assuming pooled (Batch, 1)
        d_pred_fake = self.discriminator(full_fake)

        # Losses
        g_loss_adv = self.adversarial_loss(d_pred_fake, valid)
        g_loss_l1 = self.l1_loss(y_hat, y)  # Prompting G to match target

        g_loss = self.lambda_adv * g_loss_adv + self.lambda_l1 * g_loss_l1

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        # --- Train Discriminator ---
        self.toggle_optimizer(opt_d)

        full_real = torch.cat([x, y], dim=1)
        # Detach y_hat to stop gradient to G
        full_fake_detached = torch.cat([x, y_hat.detach()], dim=1)

        d_pred_real = self.discriminator(full_real)
        d_pred_fake = self.discriminator(full_fake_detached)

        d_loss_real = self.adversarial_loss(d_pred_real, valid)
        d_loss_fake = self.adversarial_loss(d_pred_fake, fake)
        d_loss = (d_loss_real + d_loss_fake) / 2

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # Logging
        self.log("train/g_loss", g_loss, prog_bar=True)
        self.log("train/d_loss", d_loss, prog_bar=True)
        self.log("train/g_adv", g_loss_adv)
        self.log("train/g_l1", g_loss_l1)

    def validation_step(self, batch, batch_idx):
        """
        Validation step (Check G performance).
        """
        if isinstance(batch, dict):
            x = batch.get("observation")
            y = batch.get("target")
        else:
            x, y = batch

        y_hat = self.generator(x)
        val_l1 = self.l1_loss(y_hat, y)
        self.log("val/l1_loss", val_l1, prog_bar=True)
        return val_l1
