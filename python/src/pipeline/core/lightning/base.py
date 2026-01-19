"""
Base Lightning Module for the NGLab training pipeline.

Provides a template for modules with common configuration and optimization setup.
"""

from typing import Any

import pytorch_lightning as pl
import torch


class BaseModule(pl.LightningModule):
    """
    Base LightningModule with shared functionality for logging and configuration.
    """

    def __init__(self, cfg: dict[str, Any]):
        """
        Initialize the base module.

        Args:
            cfg (Dict[str, Any]): Configuration dictionary containing learning rate, etc.
        """
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.learning_rate = self.cfg.get("learning_rate", 1e-3)

    def configure_optimizers(self):
        """
        Configure the default Adam optimizer.

        Returns:
            torch.optim.Optimizer: The Adam optimizer instance.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        Abstract training step.
        """
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """
        Abstract validation step.
        """
        raise NotImplementedError
