import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class SLLightningModule(pl.LightningModule):
    """
    Module for Supervised Learning (Fine-tuning).
    """
    def __init__(self, backbone, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        self.cfg = cfg
        self.backbone = backbone
        self.head = torch.nn.Linear(cfg.get('hidden_dim', 128), cfg.get('output_dim', 1))
        
    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.get('learning_rate', 1e-3))

    def training_step(self, batch, batch_idx):
        # Batch: {observation, target}
        x = batch['observation']
        y = batch['target']
        
        pred = self(x)
        loss = F.mse_loss(pred, y) # Or CrossEntropy relative to task
        
        self.log('train/sl_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['observation']
        y = batch['target']
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log('val/sl_loss', loss)
