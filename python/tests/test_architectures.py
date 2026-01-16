
import pytest
import torch
import torch.nn as nn
from models import (
    TimeSeriesBackbone,
    MLP, RBF,
    AutoEncoder, DenoisingAE, SparseAE,
    HopfieldNetwork, RBM,
    EchoStateNetwork, ELM,
    KohonenMap, CapsuleLayer,
    RollingWindowCNN, SNN, LIFCell, SurrogateHeaviside,
    TimeGANGenerator, TimeGANDiscriminator
)
from models.diffusion_unet import DiffusionUNet1D
from pipeline.lightning.diffusion_module import DiffusionLightningModule
from pipeline.lightning.gan_module import GANLightningModule
from pipeline.lightning.base import BaseModule
from models.vae import TimeSeriesVAE, vae_loss
from pipeline.lightning.vae_module import VAELightningModule

# --- Feed-Forward & Basic Layers ---

class TestMLP:
    def test_mlp_forward(self):
        model = MLP(input_dim=10, hidden_dims=[20, 10], output_dim=5)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)

    def test_mlp_sequence(self):
        model = MLP(input_dim=10, hidden_dims=[20], output_dim=5)
        x = torch.randn(4, 30, 10)
        out = model(x, return_sequence=True)
        assert out.shape == (4, 30, 5)

class TestRBF:
    def test_rbf_forward(self):
        model = RBF(input_dim=10, num_centers=20, output_dim=5)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)

# --- AutoEncoders ---

class TestAutoEncoders:
    def test_ae(self):
        x = torch.randn(4, 10)
        ae = AutoEncoder(input_dim=10, hidden_dims=[8], latent_dim=4, output_type='prediction')
        assert ae(x).shape == (4, 10)

    def test_dae(self):
        x = torch.randn(4, 10)
        dae = DenoisingAE(input_dim=10, hidden_dims=[8], latent_dim=4, output_type='prediction')
        assert dae(x).shape == (4, 10)

    def test_sae(self):
        x = torch.randn(4, 10)
        sae = SparseAE(input_dim=10, hidden_dims=[8], latent_dim=4, output_type='embedding')
        z = sae(x)
        assert z.shape == (4, 4)
        loss = sae.sparsity_loss(z)
        assert loss > 0

class TestTimeSeriesVAE:
    @pytest.fixture
    def model_config(self):
        return {'input_dim': 5, 'latent_dim': 16, 'd_model': 32, 'seq_len': 20, 'pred_len': 5, 'encoder_type': 'lstm'}

    def test_vae_forward(self, model_config):
        model = TimeSeriesVAE(**model_config)
        x = torch.randn(8, 20, 5)
        output = model(x)
        assert output['reconstruction'].shape == (8, 5, 5)
        assert output['z'].shape == (8, 16)

# --- Energy & Associative Models ---

class TestHopfield:
    def test_hopfield(self):
        hn = HopfieldNetwork(size=10)
        patterns = torch.sign(torch.randn(2, 10))
        hn.store_patterns(patterns)
        x = patterns[0].unsqueeze(0)
        out = hn(x)
        assert out.shape == (1, 10)

class TestRBM:
    def test_rbm(self):
        rbm = RBM(visible_dim=10, hidden_dim=20, output_type='prediction')
        v = torch.bernoulli(torch.rand(4, 10))
        v_recon = rbm(v)
        assert v_recon.shape == (4, 10)

# --- Reservoir & Fast Learning ---

class TestESN:
    def test_esn(self):
        esn = EchoStateNetwork(input_dim=10, reservoir_dim=50, output_dim=5, output_type='prediction')
        x = torch.randn(4, 30, 10)
        out = esn(x)
        assert out.shape == (4, 5)

class TestELM:
    def test_elm(self):
        elm = ELM(input_dim=10, hidden_dim=50, output_dim=5, output_type='prediction')
        x_flat = torch.randn(4, 10)
        assert elm(x_flat).shape == (4, 5)

# --- Competitive & Specialized ---

class TestSOM:
    def test_som(self):
        som = KohonenMap(input_dim=10, grid_size=(5, 5))
        x = torch.randn(4, 10)
        emb = som(x)
        assert emb.shape == (4, 10)

class TestCapsule:
    def test_capsule(self):
        cap = CapsuleLayer(in_caps=8, in_dim=16, out_caps=4, out_dim=32)
        x_cap = torch.randn(4, 8, 16)
        out_cap = cap(x_cap)
        assert out_cap.shape == (4, 4, 32)
        assert torch.all(torch.norm(out_cap, dim=-1) <= 1.0)

# --- Spiking & Temporal Models ---

class TestSNN:
    def test_surrogate_gradient(self):
        x = torch.tensor([0.0], requires_grad=True)
        from models.snn import surrogate_heaviside
        y = surrogate_heaviside(x)
        y.backward()
        assert x.grad is not None

    def test_snn_forward(self):
        model = SNN(input_dim=10, hidden_dim=20, n_layers=2, output_dim=5)
        x = torch.randn(4, 30, 10)
        out = model(x)
        assert out.shape == (4, 5)

class TestCNN:
    def test_cnn_forward(self):
        model = RollingWindowCNN(input_dim=12, output_dim=1, seq_len=30, hidden_dim=64, output_type='prediction')
        x = torch.randn(2, 30, 12)
        assert model(x).shape == (2, 1)

# --- Generative & Diffusion Models ---

class TestGAN:
    def test_timegan_shapes(self):
        B, Seq, Pred, F = 2, 30, 5, 12
        gen = TimeGANGenerator(input_dim=F, output_dim=F, seq_len=Seq, pred_len=Pred, hidden_dim=16)
        disc = TimeGANDiscriminator(input_dim=F, hidden_dim=16)
        x = torch.randn(B, Seq, F)
        y_hat = gen(x)
        assert y_hat.shape == (B, Pred, F)
        assert disc(torch.cat([x, y_hat.detach()], dim=1)).shape == (B, 1)

class TestDiffusion:
    def test_diffusion_unet(self):
        B, L, F = 2, 32, 4
        cond_F = 4
        t = torch.randint(0, 1000, (B,))
        x = torch.randn(B, L, F)
        cond = torch.randn(B, L, cond_F)
        model = DiffusionUNet1D(input_dim=F+cond_F, output_dim=F, hidden_dim=16, layers=[1, 2])
        assert model(x, t, cond).shape == (B, L, F)

# --- Integration ---

class TestBackboneIntegration:
    def test_backbone_all_names(self):
        names = [
            'MLP', 'RBF', 'AE', 'DAE', 'SAE', 
            'Hopfield', 'RBM', 'ESN', 'ELM', 'SOM', 'Capsule',
            'LSTM', 'GRU', 'xLSTM', 'SNN', 'CNN'
        ]
        for name in names:
            cfg = {'name': name, 'feature_dim': 12, 'output_dim': 1}
            if name == 'Capsule':
                cfg.update({'in_caps': 8, 'in_dim': 12, 'out_caps': 4, 'out_dim': 16})
            backbone = TimeSeriesBackbone(cfg)
            assert backbone is not None
