
import pytest
import torch
import torch.nn as nn
from models.snn import SNN, LIFCell, surrogate_heaviside
from models.time_series import TimeSeriesBackbone

class TestSNN:
    @pytest.fixture
    def lif_cell(self):
        return LIFCell(input_dim=10, hidden_dim=20)

    @pytest.fixture
    def snn_model(self):
        return SNN(input_dim=10, hidden_dim=20, n_layers=2, output_dim=5)

    def test_surrogate_gradient(self):
        """Test that gradients flow through the Heaviside function."""
        x = torch.tensor([0.0], requires_grad=True)
        y = surrogate_heaviside(x)
        y.backward()
        assert x.grad is not None
        assert x.grad > 0 # At x=0, gradient should be positive (alpha)

    def test_lif_cell_forward(self, lif_cell):
        """Test LIF Cell forward pass."""
        batch_size = 4
        x = torch.randn(batch_size, 10)
        spikes, state = lif_cell(x)
        
        assert spikes.shape == (batch_size, 20)
        assert state[0].shape == (batch_size, 20) # Potential
        assert state[1].shape == (batch_size, 20) # Spikes
        
        # Spikes should be binary (0 or 1)
        assert torch.all((spikes == 0) | (spikes == 1))

    def test_snn_forward_shape(self, snn_model):
        """Test SNN forward pass output shape."""
        batch_size = 4
        seq_len = 30
        x = torch.randn(batch_size, seq_len, 10)
        
        # Test embedding output (last step)
        out = snn_model(x, return_embedding=True)
        assert out.shape == (batch_size, 20) # Hidden dim
        
        # Test prediction output
        out_pred = snn_model(x, return_embedding=False)
        assert out_pred.shape == (batch_size, 5) # Output dim
        
        # Test sequence output
        out_seq = snn_model(x, return_embedding=True, return_sequence=True)
        assert out_seq.shape == (batch_size, seq_len, 20)

    def test_snn_gradients(self, snn_model):
        """Test that gradients propagate through time (BPTT)."""
        x = torch.randn(4, 30, 10, requires_grad=True)
        y = snn_model(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_integration_backbone(self):
        """Test instantiation via TimeSeriesBackbone."""
        cfg = {
            'name': 'SNN',
            'feature_dim': 10,
            'hidden_dim': 20,
            'output_dim': 5,
            'num_layers': 2,
            'return_sequence': True
        }
        backbone = TimeSeriesBackbone(cfg)
        x = torch.randn(4, 30, 10)
        out = backbone(x)
        assert out.shape == (4, 30, 20) # Sequence of embeddings from last SNN layer
