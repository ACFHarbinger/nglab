"""
Self-Supervised Learning Module for NGLab.

Implements pretext tasks such as Masked Prediction to learn useful representations
from unlabeled financial time series data.
"""

import torch
import torch.nn.functional as F  # noqa: N812

from .base import BaseModule


class SelfSupervisedModule(BaseModule):
    """
    Module for Self-Supervised Learning tasks (e.g., Masked Prediction, Contrastive Learning).
    """

    def __init__(self, backbone, cfg):
        """
        Initialize the Self-Supervised module.

        Args:
            backbone (nn.Module): The time-series model backbone.
            cfg (Dict): Configuration parameters.
        """
        super().__init__(cfg)
        self.backbone = backbone
        self.head = torch.nn.Linear(cfg.get("hidden_dim", 128), cfg.get("input_dim", 1))
        self.mask_ratio = cfg.get("mask_ratio", 0.15)

    def forward(self, x):
        """
        Forward pass through the backbone.
        """
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        """
        Perform a self-supervised training step using Masked Prediction.
        """
        # Taking 'x' from batch (assuming tensor or dict with 'observation')
        x = batch if isinstance(batch, torch.Tensor) else batch["observation"]

        # Simple Masked Prediction Logic
        # 1. Mask input
        mask = torch.rand_like(x) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0.0  # Zero out masked values

        # 2. Forward pass
        features = self.backbone(x_masked)
        pred = self.head(features)

        # 3. Calculate Loss only on masked elements
        loss = F.mse_loss(pred[mask], x[mask])

        self.log("train/ssl_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.
        """
        x = batch if isinstance(batch, torch.Tensor) else batch["observation"]
        features = self.backbone(x)
        pred = self.head(features)
        loss = F.mse_loss(pred, x)
        self.log("val/ssl_loss", loss)
