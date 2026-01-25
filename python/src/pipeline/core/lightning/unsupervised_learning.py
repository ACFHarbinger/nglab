
"""
Unsupervised Learning Module for NGLab.

Implements deep unsupervised learning techniques such as Autoencoders
for dimensionality reduction and feature extraction from financial data.
"""

from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .base import BaseModule


class UnsupervisedModule(BaseModule):
    """
    Module for Unsupervised Learning (e.g., Autoencoder or Clustering).
    Focuses on reconstruction or density estimation.
    """

    def __init__(self, backbone: nn.Module, cfg: dict[str, Any]) -> None:
        """
        Initialize the Unsupervised module.

        Args:
            backbone (nn.Module): The time-series model backbone.
            cfg (Dict): Configuration parameters.
        """
        super().__init__(cfg)
        self.encoder = backbone
        # Simple Decoder mirroring encoder manually or learned
        self.decoder = torch.nn.Linear(
            int(cfg.get("hidden_dim", 128)), int(cfg.get("input_dim", 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (Encode and Decode).
        """
        z = self.encoder(x)
        return cast(torch.Tensor, self.decoder(z))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Perform an unsupervised training step (reconstruction).
        """
        # Batch: Tensor or Dict
        if isinstance(batch, torch.Tensor):
            x = batch
        elif isinstance(batch, dict):
            x = batch["observation"]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        reconstruction = self(x)
        loss = F.mse_loss(reconstruction, x)

        self.log("train/recon_loss", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """
        Perform a validation step.
        """
        if isinstance(batch, torch.Tensor):
            x = batch
        elif isinstance(batch, dict):
            x = batch["observation"]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        reconstruction = self(x)
        loss = F.mse_loss(reconstruction, x)
        self.log("val/recon_loss", loss)
