import pytest
import torch
from models.cnn import RollingWindowCNN

def test_cnn_prediction_mode():
    """Test RollingWindowCNN in prediction mode."""
    B, L, F = 2, 30, 12
    hidden_dim = 64
    
    model = RollingWindowCNN(input_dim=F, output_dim=1, seq_len=L, hidden_dim=hidden_dim, output_type='prediction')
    x = torch.randn(B, L, F)
    out = model(x)
    
    assert out.shape == (B, 1)

def test_cnn_embedding_mode():
    """Test RollingWindowCNN in embedding mode."""
    B, L, F = 2, 30, 12
    hidden_dim = 64
    
    model = RollingWindowCNN(input_dim=F, output_dim=1, seq_len=L, hidden_dim=hidden_dim, output_type='embedding')
    x = torch.randn(B, L, F)
    out = model(x)
    
    # Embedding output is the result of fc1 which outputs (B, hidden_dim)
    assert out.shape == (B, hidden_dim)
