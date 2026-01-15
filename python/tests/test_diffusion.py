import pytest
import torch
from models.diffusion_unet import DiffusionUNet1D
from pipeline.lightning.diffusion_module import DiffusionLightningModule

def test_diffusion_unet_forward():
    """Test the forward pass of DiffusionUNet1D."""
    B, L, F = 2, 32, 4
    cond_F = 4
    t = torch.randint(0, 1000, (B,))
    x = torch.randn(B, L, F)
    cond = torch.randn(B, L, cond_F)
    
    # Input dim = F + cond_F because of concatenation logic in forward (if assumed)
    # The actual implementation calls `x = torch.cat([x, cond], dim=-1)`
    # So input_dim to UNet init should be F + cond_F?
    # Let's check implementation behavior: 
    # __init__ takes input_dim. forward takes (x, t, cond).
    # Inside forward: `x = torch.cat([x, cond], dim=-1)`. Then `self.init_conv(x)`.
    # So init_conv input channels = F + cond_F.
    # Therefore we must initialize UNet with input_dim = F + cond_F.
    
    model = DiffusionUNet1D(input_dim=F+cond_F, output_dim=F, hidden_dim=16, layers=[1, 2])
    out = model(x, t, cond)
    
    assert out.shape == (B, L, F)

def test_diffusion_module_training_step():
    """Test the training step of DiffusionLightningModule."""
    B, L, F = 2, 32, 4
    # Condition has same dim as features for simplicity here
    model = DiffusionUNet1D(input_dim=F*2, output_dim=F, hidden_dim=16, layers=[1])
    
    cfg = {
        'timesteps': 100,
        'pred_len': 32
    }
    module = DiffusionLightningModule(model, cfg)
    
    batch = {
        'observation': torch.randn(B, L, F), # Cond (F)
        'target': torch.randn(B, L, F)      # Target (F) -> x_start
    }
    
    loss = module.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0 # scalar
    
def test_diffusion_module_sampling():
    """Test the sampling method of DiffusionLightningModule."""
    B, L, F = 2, 32, 4
    model = DiffusionUNet1D(input_dim=F*2, output_dim=F, hidden_dim=16, layers=[1])
    
    cfg = {
        'timesteps': 10, # small for speed
        'pred_len': 32
    }
    module = DiffusionLightningModule(model, cfg)
    
    cond = torch.randn(B, L, F)
    sample = module.sample(cond)
    
    assert sample.shape == (B, L, F)
