import pytest
import torch
from models.gan_networks import TimeGANGenerator, TimeGANDiscriminator
from pipeline.lightning.gan_module import GANLightningModule
from pipeline.lightning.base import BaseModule

def test_timegan_networks_shapes():
    """Test forward pass shapes of Generator and Discriminator."""
    B, Seq, Pred, F = 2, 30, 5, 12
    
    # 1. Generator
    gen = TimeGANGenerator(input_dim=F, output_dim=F, seq_len=Seq, pred_len=Pred, hidden_dim=16)
    x = torch.randn(B, Seq, F)
    y_hat = gen(x)
    
    assert y_hat.shape == (B, Pred, F)
    
    # 2. Discriminator
    disc = TimeGANDiscriminator(input_dim=F, hidden_dim=16)
    # D takes (B, Total_Seq, F)
    full_seq = torch.cat([x, y_hat.detach()], dim=1)
    d_score = disc(full_seq)
    
    assert d_score.shape == (B, 1)

def test_gan_module_config():
    """Test GANLightningModule configuration and inheritance."""
    B, Seq, Pred, F = 2, 30, 5, 12
    
    cfg = {
        'lambda_adv': 1.0,
        'lambda_l1': 100.0,
        'lr_g': 1e-3,
        'lr_d': 1e-3,
        'learning_rate': 1e-3
    }
    
    gen = TimeGANGenerator(input_dim=F, output_dim=F, seq_len=Seq, pred_len=Pred, hidden_dim=16)
    disc = TimeGANDiscriminator(input_dim=F, hidden_dim=16)
    
    module = GANLightningModule(gen, disc, cfg)
    
    # Inheritance check
    assert isinstance(module, BaseModule)
    
    # Config check
    assert module.cfg == cfg
    
    # Optimizer check
    opts, _ = module.configure_optimizers()
    assert len(opts) == 2
