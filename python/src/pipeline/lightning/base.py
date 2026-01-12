import pytorch_lightning as pl
import torch
from typing import Dict, Any

class BaseModule(pl.LightningModule):
    """
    Base LightningModule with shared functionality for logging and configuration.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.learning_rate = self.cfg.get('learning_rate', 1e-3)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
