"""
Generative Adversarial Network (GAN) Module for Time Series Prediction.
"""

from typing import Any, Dict, List, Tuple, Union, cast

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Optimizer

from .base import BaseModule


class GANLightningModule(BaseModule):
    """
    Lightning Module for Time Series GAN (e.g., TimeGAN style or conditional forecasting GAN).

    Structure:
    - Generator: Takes history (X) -> Predicts future (Y_hat).
    - Discriminator: Takes full sequence (X + Y) -> Predicts Real/Fake.
    """

    def __init__(
        self, generator: nn.Module, discriminator: nn.Module, cfg: Dict[str, Any]
    ) -> None:
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
        self.lambda_adv = float(cfg.get("lambda_adv", 1.0))
        self.lambda_l1 = float(cfg.get("lambda_l1", 100.0))
        self.lr_g = float(cfg.get("lr_g", 2e-4))
        self.lr_d = float(cfg.get("lr_d", 2e-4))
        self.b1 = float(cfg.get("beta1", 0.5))
        self.b2 = float(cfg.get("beta2", 0.999))

        # Buffer for input/target concatenation if needed
        self.automatic_optimization = False  # We handle G and D steps manually

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (Generator prediction).
        """
        return cast(torch.Tensor, self.generator(x))

    def configure_optimizers(self) -> Any:  # Returning Any to avoid complexity with PyTorch Lightning types
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

    def training_step(self, batch: Any, batch_idx: int) -> None:  # type: ignore[override]
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

        if x is None or y is None:
            raise ValueError("Training batch must contain observation and target")

        # Optimizers
        optimizers = self.optimizers()
        if isinstance(optimizers, list):
             # When return is List[LightningOptimizer], but typing says List[Optimizer]
             # Lightining returns LightningOptimizer wrapper. Using untyped access or ignoring.
             # Actually self.optimizers() usually returns a single optimizer or list.
             # With Manual optimization, we access them via indices usually or just tuple unpacking if we know count.
             opt_g, opt_d = optimizers[0], optimizers[1]
        else:
             # Should not happen given configure_optimizers returns list
             raise RuntimeError("Expected list of optimizers")

        device = cast(torch.device, self.device)

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
        
        # NOTE: If Feature dims don't match, we assume they do for TS prediction tasks.
        full_fake = torch.cat([x, y_hat], dim=1)

        # Adversarial ground truth
        valid = torch.ones((x.size(0), 1), device=device)
        fake = torch.zeros((x.size(0), 1), device=device)

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

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Validation step (Check G performance).
        """
        if isinstance(batch, dict):
            x = batch.get("observation")
            y = batch.get("target")
        else:
            x, y = batch

        if x is None or y is None:
             raise ValueError("Validation batch must contain observation and target")

        y_hat = self.generator(x)
        val_l1 = self.l1_loss(y_hat, y)
        self.log("val/l1_loss", val_l1, prog_bar=True)
        return cast(torch.Tensor, val_l1)
