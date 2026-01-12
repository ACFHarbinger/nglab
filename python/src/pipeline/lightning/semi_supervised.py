"""
Semi-Supervised Learning Module for NGLab.

Implements techniques like pseudo-labeling and consistency regularization to leverage
both labeled and unlabeled data for training.
"""
import torch
import torch.nn.functional as F
from .base import BaseModule

class SemiSupervisedModule(BaseModule):
    """
    Module for Semi-Supervised Learning (e.g., FixMatch using Pseudo-labeling).
    Combines labeled loss with consistency regularization on unlabeled data.
    """
    def __init__(self, backbone, cfg):
        """
        Initialize the Semi-Supervised module.

        Args:
            backbone (nn.Module): The time-series model backbone.
            cfg (Dict): Configuration parameters.
        """
        super().__init__(cfg)
        self.backbone = backbone
        self.head = torch.nn.Linear(cfg.get('hidden_dim', 128), cfg.get('num_classes', 2))
        self.threshold = cfg.get('threshold', 0.95)
        self.lambda_u = cfg.get('lambda_u', 1.0)

    def forward(self, x):
        """
        Forward pass through the backbone and head.
        """
        return self.head(self.backbone(x))

    def training_step(self, batch, batch_idx):
        """
        Perform a semi-supervised training step.
        """
        # Expecting batch to have labeled and unlabeled data
        x_labeled, y_labeled = batch.get('labeled', (None, None))
        x_unlabeled = batch.get('unlabeled', None)
        
        loss = 0.0
        
        # Supervised Loss
        if x_labeled is not None:
            logits_labeled = self(x_labeled)
            loss_s = F.cross_entropy(logits_labeled, y_labeled)
            self.log('train/loss_s', loss_s)
            loss += loss_s
            
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
            loss_u = (F.cross_entropy(logits_u_strong, targets_u, reduction='none') * mask).mean()
            self.log('train/loss_u', loss_u)
            loss += self.lambda_u * loss_u

        self.log('train/total_loss', loss)
        return loss
