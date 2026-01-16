
import pytest
import torch
import torch.nn as nn
from models import (
    TimeSeriesBackbone,
    MLP, RBF,
    AutoEncoder, DenoisingAE, SparseAE,
    HopfieldNetwork, RBM,
    EchoStateNetwork, ELM,
    KohonenMap, CapsuleLayer
)

class TestNewArchitectures:
    
    def test_mlp(self):
        # Default is prediction if called directly with my implementation
        model = MLP(input_dim=10, hidden_dims=[20, 10], output_dim=5)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)

    def test_rbf(self):
        model = RBF(input_dim=10, num_centers=20, output_dim=5)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)

    def test_autoencoders(self):
        x = torch.randn(4, 10)
        
        # Standard AE - now returns just one value by default based on output_type
        ae = AutoEncoder(input_dim=10, hidden_dims=[8], latent_dim=4, output_type='prediction')
        assert ae(x).shape == (4, 10)
        
        # Denoising AE
        dae = DenoisingAE(input_dim=10, hidden_dims=[8], latent_dim=4, output_type='prediction')
        assert dae(x).shape == (4, 10)
        
        # Sparse AE
        sae = SparseAE(input_dim=10, hidden_dims=[8], latent_dim=4, output_type='embedding')
        z = sae(x)
        assert z.shape == (4, 4)
        loss = sae.sparsity_loss(z)
        assert loss > 0

    def test_energy_models(self):
        # Hopfield
        hn = HopfieldNetwork(size=10)
        patterns = torch.sign(torch.randn(2, 10))
        hn.store_patterns(patterns)
        x = patterns[0].unsqueeze(0)
        out = hn(x)
        assert out.shape == (1, 10)
        
        # RBM
        rbm = RBM(visible_dim=10, hidden_dim=20, output_type='prediction') # Returns reconstruction
        v = torch.bernoulli(torch.rand(4, 10))
        v_recon = rbm(v)
        assert v_recon.shape == (4, 10)

    def test_reservoir_models(self):
        # ESN
        esn = EchoStateNetwork(input_dim=10, reservoir_dim=50, output_dim=5, output_type='prediction')
        x = torch.randn(4, 30, 10)
        out = esn(x)
        assert out.shape == (4, 5)
        out_seq = esn(x, return_sequence=True)
        assert out_seq.shape == (4, 30, 5)
        
        # ELM
        elm = ELM(input_dim=10, hidden_dim=50, output_dim=5, output_type='prediction')
        x_flat = torch.randn(4, 10)
        assert elm(x_flat).shape == (4, 5)

    def test_specialized_models(self):
        # Kohonen SOM - returns BMU weights as embedding
        som = KohonenMap(input_dim=10, grid_size=(5, 5))
        x = torch.randn(4, 10)
        emb = som(x)
        assert emb.shape == (4, 10)
        
        # Capsule Layer
        cap = CapsuleLayer(in_caps=8, in_dim=16, out_caps=4, out_dim=32)
        x_cap = torch.randn(4, 8, 16)
        out_cap = cap(x_cap)
        assert out_cap.shape == (4, 4, 32)
        norms = torch.norm(out_cap, dim=-1)
        assert torch.all(norms <= 1.0)

    def test_backbone_integration(self):
        # MLP
        cfg_mlp = {'name': 'MLP', 'feature_dim': 10, 'hidden_dims': [20], 'output_dim': 5, 'output_type': 'prediction'}
        backbone_mlp = TimeSeriesBackbone(cfg_mlp)
        x = torch.randn(4, 10)
        assert backbone_mlp(x).shape == (4, 5)
        
        # ESN
        cfg_esn = {'name': 'ESN', 'feature_dim': 10, 'hidden_dim': 50, 'output_dim': 5, 'return_sequence': True, 'output_type': 'prediction'}
        backbone_esn = TimeSeriesBackbone(cfg_esn)
        x_seq = torch.randn(4, 30, 10)
        assert backbone_esn(x_seq).shape == (4, 30, 5)
        
        # ELM
        cfg_elm = {'name': 'ELM', 'feature_dim': 10, 'hidden_dim': 50, 'output_dim': 5, 'output_type': 'prediction'}
        backbone_elm = TimeSeriesBackbone(cfg_elm)
        assert backbone_elm(x).shape == (4, 5)
