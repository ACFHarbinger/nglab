
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

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        """
        Initialize the base module.

        Args:
            cfg (Dict[str, Any], optional): Configuration dictionary containing learning rate, etc.
        """
        super().__init__()
        self.save_hyperparameters()
        # Ensure cfg is set and accessible
        self.cfg = cfg or {}
        self.learning_rate = float(self.cfg.get("learning_rate", 1e-3))

    def configure_optimizers(self) -> Any:
        """
        Configure the default Adam optimizer.

        Returns:
            Any: Optimizer or dict containing optimizer and scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """
        Abstract training step.
        """
        raise NotImplementedError

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        """
        Abstract validation step.
        """
        raise NotImplementedError
