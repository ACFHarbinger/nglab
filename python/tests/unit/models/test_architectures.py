import torch
from torch import nn

from python.src.models import TimeSeriesBackbone
from python.src.models import (
    DCIGN,
    DNC,
    ELM,
    LVQ,
    MLP,
    NTM,
    PINN,
    RBF,
    RBM,
    SNN,
    VAE,
    AttentionNetwork,
    AutoDeconvNet,
    AutoEncoder,
    BoltzmannMachine,
    CapsuleLayer,
    DeconvNet,
    DeepBeliefNetwork,
    DeepConvNet,
    DeepResNet,
    DenoisingAE,
    EchoStateNetwork,
    HopfieldNetwork,
    KohonenMap,
    LiquidStateMachine,
    MarkovChain,
    NeuralODE,
    NormalizingFlow,
    Perceptron,
    RollingWindowCNN,
    SparseAE,
    StackedAutoEncoder,
)
from python.src.models.general.pinn import pinn_loss

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


class TestPerceptron:
    def test_perceptron(self):
        model = Perceptron(input_dim=10, output_dim=5)
        x = torch.randn(4, 10)
        assert model(x).shape == (4, 5)


class TestMarkovChain:
    def test_markov_chain(self):
        model = MarkovChain(num_states=10)
        # MarkovChain expects state probabilities or one-hot
        x = torch.randn(4, 10).softmax(dim=-1)
        assert model(x).shape == (4, 10)


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
        ae = AutoEncoder(
            input_dim=10, hidden_dims=[8], latent_dim=4, output_type="prediction"
        )
        assert ae(x).shape == (4, 10)

    def test_dae(self):
        x = torch.randn(4, 10)
        dae = DenoisingAE(
            input_dim=10, hidden_dims=[8], latent_dim=4, output_type="prediction"
        )
        assert dae(x).shape == (4, 10)

    def test_sae(self):
        x = torch.randn(4, 10)
        sae = SparseAE(
            input_dim=10, hidden_dims=[8], latent_dim=4, output_type="embedding"
        )
        z = sae(x)
        assert z.shape == (4, 4)
        loss = sae.sparsity_loss(z)
        loss = sae.sparsity_loss(z)
        assert loss > 0

    def test_stacked_ae(self):
        x = torch.randn(4, 10)
        st_ae = StackedAutoEncoder(layer_sizes=[10, 8, 4], output_type="prediction")
        # 10 -> 8 -> 4 (latent) -> 8 -> 10 (recon)
        assert st_ae(x).shape == (4, 10)

        # Test sequences
        x_seq = torch.randn(4, 5, 10)
        assert st_ae(x_seq, return_sequence=True).shape == (4, 5, 10)


class TestVAE:
    def test_vae_forward(self, vae_config):
        model = VAE(**vae_config)
        # Match vae_config: seq_len=30, input_dim=10
        x = torch.randn(8, 30, 10)
        output = model(x)
        assert output["reconstruction"].shape == (8, 5, 10)
        assert output["z"].shape == (8, 16)


# --- Energy & Associative Models ---


class TestHopfield:
    def test_hopfield(self):
        hn = HopfieldNetwork(size=10)
        patterns = torch.sign(torch.randn(2, 10))
        hn.store_patterns(patterns)
        x = patterns[0].unsqueeze(0)
        out = hn(x)
        assert out.shape == (1, 10)


class TestBoltzmann:
    def test_bm(self):
        model = BoltzmannMachine(num_units=10)
        x = torch.bernoulli(torch.rand(4, 10))
        out = model(x)
        assert out.shape == (4, 10)


class TestRBM:
    def test_rbm(self):
        rbm = RBM(visible_dim=10, hidden_dim=20, output_type="prediction")
        v = torch.bernoulli(torch.rand(4, 10))
        v_recon = rbm(v)
        assert v_recon.shape == (4, 10)


class TestDBN:
    def test_dbn(self):
        model = DeepBeliefNetwork(layer_sizes=[10, 20, 5])
        x = torch.randn(4, 10)
        assert model(x).shape == (4, 5)


# --- Reservoir & Fast Learning ---


class TestESN:
    def test_esn(self):
        esn = EchoStateNetwork(
            input_dim=10, reservoir_dim=50, output_dim=5, output_type="prediction"
        )
        x = torch.randn(4, 30, 10)
        out = esn(x)
        assert out.shape == (4, 5)


class TestELM:
    def test_elm(self):
        elm = ELM(input_dim=10, hidden_dim=50, output_dim=5, output_type="prediction")
        x_flat = torch.randn(4, 10)
        assert elm(x_flat).shape == (4, 5)


class TestLiquidStateMachine:
    def test_lsm_forward(self):
        model = LiquidStateMachine(input_dim=12, liquid_size=100, output_dim=1)
        x = torch.randn(2, 30, 12)
        assert model(x).shape == (2, 1)


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


# --- Convolutional & Residual Variants ---


class TestConvVariants:
    def test_dcn(self):
        model = DeepConvNet(input_dim=12, hidden_channels=[32, 64], output_dim=1)
        x = torch.randn(2, 30, 12)
        assert model(x).shape == (2, 1)

    def test_deconv(self):
        model = DeconvNet(input_dim=64, hidden_channels=[128, 64], output_dim=12)
        x = torch.randn(2, 64)
        assert model(x).shape == (2, 12)


class TestDeconvNet:
    def test_auto_deconv(self):
        model = AutoDeconvNet(input_dim=12, latent_dim=64, hidden_channels=[32, 64])
        x = torch.randn(2, 30, 12)
        assert model(x).shape == (2, 12)


class TestDCIGN:
    def test_dcign_forward(self):
        model = DCIGN(input_dim=12, latent_dim=64, hidden_channels=[32, 64])
        x = torch.randn(2, 30, 12)
        assert model(x).shape == (2, 12)

    def test_dcign_codes(self):
        model = DCIGN(
            input_dim=12,
            latent_dim=64,
            hidden_channels=[32, 64],
            output_type="embedding",
        )
        x = torch.randn(2, 30, 12)
        assert model(x).shape == (2, 64)


class TestResNet:
    def test_resnet_forward(self):
        model = DeepResNet(input_dim=12, hidden_dim=64, num_blocks=2, output_dim=1)
        x = torch.randn(2, 30, 12)
        assert model(x).shape == (2, 1)


# --- Spiking & Temporal Models ---


class TestSNN:
    def test_surrogate_gradient(self):
        x = torch.tensor([0.0], requires_grad=True)
        from python.src.models.spiking import surrogate_heaviside

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
        model = RollingWindowCNN(
            input_dim=12,
            output_dim=1,
            seq_len=30,
            hidden_dim=64,
            output_type="prediction",
        )
        x = torch.randn(2, 30, 12)
        assert model(x).shape == (2, 1)


# --- Memory-Augmented & Attention ---


class TestMemoryAugmented:
    def test_dnc_forward(self):
        model = DNC(
            input_dim=12,
            hidden_dim=64,
            memory_size=16,
            memory_dim=16,
            num_reads=2,
            output_dim=1,
        )
        x = torch.randn(2, 10, 12)
        assert model(x).shape == (2, 1)

    def test_ntm_forward(self):
        model = NTM(
            input_dim=12,
            hidden_dim=64,
            memory_size=16,
            memory_dim=16,
            num_reads=1,
            num_writes=1,
            output_dim=1,
        )
        x = torch.randn(2, 10, 12)
        assert model(x).shape == (2, 1)


class TestAttention:
    def test_attention_forward(self):
        model = AttentionNetwork(
            input_dim=12, d_model=64, num_layers=2, num_heads=4, d_ff=128, output_dim=1
        )
        x = torch.randn(2, 10, 12)
        assert model(x).shape == (2, 1)


# --- Generative Models ---


class TestNormalizingFlow:
    def test_flow_forward(self):
        input_dim = 10
        model = NormalizingFlow(input_dim=input_dim, num_layers=2, hidden_dim=20)
        x = torch.randn(4, 10)
        z, log_det = model(x)
        assert z.shape == (4, 10)
        assert log_det.shape == (4,)

    def test_flow_inverse(self):
        input_dim = 10
        model = NormalizingFlow(input_dim=input_dim, num_layers=4, hidden_dim=20)
        x = torch.randn(4, 10)
        z, _ = model(x)
        x_recon = model.inverse(z)
        assert torch.allclose(x, x_recon, atol=1e-5)

    def test_flow_sampling(self):
        input_dim = 10
        model = NormalizingFlow(input_dim=input_dim, num_layers=2, hidden_dim=20)
        samples = model.sample(num_samples=5, device=torch.device("cpu"))
        assert samples.shape == (5, 10)

    def test_flow_sequence(self):
        # Test handling of flattened sequence
        input_dim = 5
        seq_len = 5
        model = NormalizingFlow(input_dim=input_dim, seq_len=seq_len, num_layers=2)
        x = torch.randn(3, seq_len, input_dim)
        z, _log_det = model(x)
        assert z.shape == (3, seq_len * input_dim)  # Latent is flattened

        x_recon = model.inverse(z)
        assert x_recon.shape == (3, seq_len, input_dim)
        assert torch.allclose(x, x_recon, atol=1e-5)


class TestNeuralODE:
    def test_node_forward(self):
        model = NeuralODE(input_dim=2, hidden_dim=10, time_steps=5, horizon=1.0)
        x = torch.randn(4, 2)
        out = model(x)
        assert out.shape == (4, 5, 1)  # (batch, steps, output_dim=1 default)

    def test_node_solver(self):
        # Test if simple decay exponential solution is valid
        # dy/dt = -y => y(t) = y0 * exp(-t)
        from python.src.models.general import odesolve

        class Decay(nn.Module):
            def forward(self, t, y):
                return -y

        y0 = torch.tensor([[1.0]])
        t = torch.linspace(0, 1.0, 10)  # More steps for accuracy
        y = odesolve(Decay(), y0, t)

        expected_last = torch.exp(torch.tensor(-1.0))
        assert torch.allclose(y[-1, 0, 0], expected_last, atol=1e-4)


class TestPINN:
    def test_pinn_gradient(self):
        model = PINN(input_dim=1, output_dim=1)
        x = torch.tensor([[1.0], [2.0]], requires_grad=True)
        u = model(x)
        grad = model.gradient(u, x)
        assert grad.shape == (2, 1)

    def test_pinn_loss(self):
        model = PINN(input_dim=1, output_dim=1)
        x = torch.randn(4, 1, requires_grad=True)
        u = model(x)
        # Fake PDE residual: du/dx = 0
        grad = model.gradient(u, x)
        loss_dict = pinn_loss(u, torch.randn_like(u), grad)
        assert "total_loss" in loss_dict


# --- Integration ---


class TestBackboneIntegration:
    def test_backbone_all_names(self):
        names = [
            "MLP",
            "RBF",
            "AE",
            "DAE",
            "DAE",
            "SAE",
            "StackedAE",
            "Hopfield",
            "RBM",
            "ESN",
            "ELM",
            "SOM",
            "Capsule",
            "LSTM",
            "GRU",
            "xLSTM",
            "SNN",
            "CNN",
            "Perceptron",
            "MarkovChain",
            "BM",
            "DBN",
            "DCN",
            "Deconv",
            "DCIGN",
            "LSM",
            "ResNet",
            "DNC",
            "NTM",
            "Attention",
            "LVQ",
        ]
        for name in names:
            cfg = {"name": name, "feature_dim": 12, "output_dim": 1}
            if name == "Capsule":
                cfg.update({"in_caps": 8, "in_dim": 12, "out_caps": 4, "out_dim": 16})
            backbone = TimeSeriesBackbone(cfg)
            assert backbone is not None

    def test_backbone_new_models_forward(self):
        # Smoke test for forward pass of some of the complex new models via backbone
        complex_models = [
            "LSM",
            "ResNet",
            "DNC",
            "NTM",
            "Attention",
            "DCIGN",
            "DCN",
            "NODE",
            "PINN",
        ]
        for name in complex_models:
            cfg = {"name": name, "feature_dim": 12, "output_dim": 12, "seq_len": 5}
            backbone = TimeSeriesBackbone(cfg)
            x = torch.randn(2, 20, 12)
            out = backbone(x)
            assert out.shape[0] == 2
            assert out.shape[-1] == 12

    def test_backbone_flow(self):
        cfg = {"name": "Flow", "feature_dim": 10, "num_layers": 2}
        backbone = TimeSeriesBackbone(cfg)
        x = torch.randn(4, 10)
        # Flow returns (z, log_det) tuple
        z, log_det = backbone(x)
        assert z.shape == (4, 10)
        assert log_det.shape == (4,)


class TestLVQ:
    def test_lvq_forward(self):
        model = LVQ(input_dim=10, num_classes=3, output_type="prediction")
        x = torch.randn(4, 10)
        # Check prediction shape
        out = model(x)
        assert out.shape == (4, 1)

    def test_lvq_embedding(self):
        model = LVQ(input_dim=10, num_classes=3, output_type="embedding")
        x = torch.randn(4, 10)
        # Embedding returns distances to prototypes
        out = model(x, return_embedding=True)
        # num_prototypes = 3 * 1 = 3
        assert out.shape == (4, 3)

    def test_lvq_training_step(self):
        model = LVQ(input_dim=10, num_classes=2)
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4, 1))
        loss = model.training_step(x, y)
        assert loss is not None
        assert (
            loss.grad_fn is not None or loss.requires_grad
        )  # Ensure it's part of graph
