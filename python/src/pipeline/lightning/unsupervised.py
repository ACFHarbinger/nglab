import torch
import torch.nn.functional as F
from .base import BaseModule

class UnsupervisedModule(BaseModule):
    """
    Module for Unsupervised Learning (e.g., Autoencoder or Clustering).
    Focuses on reconstruction or density estimation.
    """
    def __init__(self, backbone, cfg):
        super().__init__(cfg)
        self.encoder = backbone
        # Simple Decoder mirroring encoder manually or learned
        self.decoder = torch.nn.Linear(cfg.get('hidden_dim', 128), cfg.get('input_dim', 1))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        x = batch if isinstance(batch, torch.Tensor) else batch['observation']
        
        reconstruction = self(x)
        loss = F.mse_loss(reconstruction, x)
        
        self.log('train/recon_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch if isinstance(batch, torch.Tensor) else batch['observation']
        reconstruction = self(x)
        loss = F.mse_loss(reconstruction, x)
        self.log('val/recon_loss', loss)
