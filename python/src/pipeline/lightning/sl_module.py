"""
Supervised Learning Module for NGLab.

Implements standard supervised fine-tuning of pre-trained backbones
for specific downstream prediction tasks.
"""
import torch
import torch.nn.functional as F
from .base import BaseModule

class SLLightningModule(BaseModule):
    """
    Module for Supervised Learning (Fine-tuning).
    """
    def __init__(self, backbone, cfg):
        """
        Initialize the Supervised module.

        Args:
            backbone (nn.Module): The pre-trained time-series model backbone.
            cfg (Dict): Configuration parameters.
        """
        super().__init__(cfg)
        self.save_hyperparameters(ignore=['backbone'])
        self.backbone = backbone
        self.head = torch.nn.Linear(cfg.get('hidden_dim', 128), cfg.get('output_dim', 1))
        
    def forward(self, x):
        """
        Forward pass through the backbone and head.
        """
        feat = self.backbone(x)
        return self.head(feat)


    def training_step(self, batch, batch_idx):
        """
        Perform a supervised training step.
        """
        # Batch: {observation, target}
        x = batch['observation']
        y = batch['target']
        
        pred = self(x)
        loss = F.mse_loss(pred, y) # Or CrossEntropy relative to task
        
        self.log('train/sl_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.
        """
        x = batch['observation']
        y = batch['target']
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log('val/sl_loss', loss)
