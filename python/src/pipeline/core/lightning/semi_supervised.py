"""
Semi-Supervised Learning Module for NGLab.

Implements techniques like pseudo-labeling and consistency regularization to leverage
both labeled and unlabeled data for training.
"""

from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .base import BaseModule


class SemiSupervisedModule(BaseModule):
    """
    Module for Semi-Supervised Learning (e.g., FixMatch using Pseudo-labeling).
    Combines labeled loss with consistency regularization on unlabeled data.
    """

    def __init__(self, backbone: nn.Module, cfg: dict[str, Any]) -> None:
        """
        Initialize the Semi-Supervised module.

        Args:
            backbone (nn.Module): The time-series model backbone.
            cfg (Dict): Configuration parameters.
        """
        super().__init__(cfg)
        self.backbone = backbone
        self.head = torch.nn.Linear(
            int(cfg.get("hidden_dim", 128)), int(cfg.get("num_classes", 2))
        )
        self.threshold = float(cfg.get("threshold", 0.95))
        self.lambda_u = float(cfg.get("lambda_u", 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone and head.
        """
        return cast(torch.Tensor, self.head(self.backbone(x)))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Perform a semi-supervised training step.
        """
        if not isinstance(batch, dict):
            raise ValueError("SemiSupervisedModule requires batch to be a dictionary")

        # Expecting batch to have labeled and unlabeled data
        x_labeled, y_labeled = batch.get("labeled", (None, None))
        x_unlabeled = batch.get("unlabeled", None)

        loss: torch.Tensor = torch.tensor(0.0, device=cast(torch.device, self.device))

        # Supervised Loss
        if x_labeled is not None:
            logits_labeled = self(x_labeled)
            loss_s = F.cross_entropy(logits_labeled, y_labeled)
            self.log("train/loss_s", loss_s)
            loss = loss + loss_s  # Ensure distinct tensor creation

        # Unsupervised (Consistency) Loss
        if x_unlabeled is not None:
            # Pseudo-labeling
            with torch.no_grad():
                logits_u = self(x_unlabeled)
                probs_u = torch.softmax(logits_u, dim=-1)
                max_probs, targets_u = torch.max(probs_u, dim=-1)
                mask = max_probs.ge(self.threshold).float()

            # Re-compute logits (e.g. with augmentation/dropout enabled)
            # Here assuming simple consistency
            logits_u_strong = self(x_unlabeled)
            loss_u_elements = F.cross_entropy(
                logits_u_strong, targets_u, reduction="none"
            )
            loss_u = (loss_u_elements * mask).mean()

            self.log("train/loss_u", loss_u)
            loss = loss + (self.lambda_u * loss_u)

        self.log("train/total_loss", loss)
        return loss
