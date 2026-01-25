"""
Deep Learning Model Factory.
"""

from __future__ import annotations

from typing import Any

from torch import nn

__all__ = ["DEEP_MODEL_NAMES", "create_deep_model"]

from .attention import AttentionNetwork, NSTransformer
from .autoencoders import (
    VAE,
    AutoEncoder,
    DenoisingAE,
    SparseAE,
    StackedAutoEncoder,
)
from .competitive import LVQ, KohonenMap
from .convolutional import (
    DCIGN,
    AutoDeconvNet,
    CapsuleLayer,
    DeconvNet,
    DeepConvNet,
    DeepResNet,
    RollingWindowCNN,
)
from .general import (
    ELM,
    MLP,
    NeuralODE,
    Perceptron,
    PINN,
    RBF,
)
from .memory import DNC, NTM
from .probabilistic import (
    RBM,
    BoltzmannMachine,
    DeepBeliefNetwork,
    HopfieldNetwork,
    MarkovChain,
    NormalizingFlow,
)
from .recurrent import (
    GRU,
    LSTM,
    EchoStateNetwork,
    LiquidStateMachine,
    TSMamba,
    xLSTM,
)
from .spiking import SNN

# List of deep model names
DEEP_MODEL_NAMES: list[str] = [
    "NSTransformer",
    "Mamba",
    "LSTM",
    "GRU",
    "xLSTM",
    "SNN",
    "MLP",
    "RBF",
    "AE",
    "DAE",
    "SAE",
    "StackedAE",
    "Hopfield",
    "RBM",
    "ESN",
    "ELM",
    "SOM",
    "Capsule",
    "CNN",
    "Perceptron",
    "MarkovChain",
    "BM",
    "DBN",
    "DCN",
    "Deconv",
    "AutoDeconv",
    "DCIGN",
    "LSM",
    "ResNet",
    "DNC",
    "NTM",
    "Attention",
    "Flow",
    "NODE",
    "LVQ",
    "PINN",
    "VAE",
]


def create_deep_model(model_name: str, cfg: dict[str, Any]) -> nn.Module | None:
    """
    Factory function to create deep learning models.

    Args:
        model_name: Name of the model to create.
        cfg: Configuration dictionary.

    Returns:
        Instantiated model or None if not a deep model.
    """
    if model_name == "NSTransformer":
        return NSTransformer(
            pred_len=int(cfg.get("pred_len", 1)),
            seq_len=int(cfg.get("seq_len", 30)),
            input_dim=int(cfg.get("feature_dim", 12)),
            embed_dim=int(cfg.get("embed_dim", 64)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            output_dim=int(cfg.get("output_dim", 64)),
            learner_dims=cfg.get("learner_dims", [64]),
        )
    elif model_name == "Mamba":
        return TSMamba(
            input_dim=int(cfg.get("feature_dim", 12)),
            output_dim=1,
            d_model=int(cfg.get("hidden_dim", 128)),
            n_layers=int(cfg.get("num_layers", 2)),
            forecast_horizon=int(cfg.get("pred_len", 1)),
            output_type=cfg.get("output_type", "embedding"),
        )
    elif model_name == "LSTM":
        return LSTM(
            input_dim=int(cfg.get("feature_dim", 12)),
            output_dim=int(cfg.get("output_dim", 1)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            n_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.0)),
            output_type=cfg.get("output_type", "embedding"),
            apply_softmax=cfg.get("probabilistic", False),
        )
    elif model_name == "GRU":
        return GRU(
            input_dim=int(cfg.get("feature_dim", 12)),
            output_dim=int(cfg.get("output_dim", 1)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            n_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.0)),
            output_type=cfg.get("output_type", "embedding"),
        )
    elif model_name == "xLSTM":
        return xLSTM(
            input_dim=int(cfg.get("feature_dim", 12)),
            output_dim=1,
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            n_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.0)),
            output_type=cfg.get("output_type", "embedding"),
            cell_type=cfg.get("cell_type", "slstm"),
            num_heads=int(cfg.get("num_heads", 4)),
        )
    elif model_name == "SNN":
        return SNN(
            input_dim=int(cfg.get("feature_dim", 12)),
            output_dim=int(cfg.get("output_dim", 1)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            n_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.0)),
            output_type=cfg.get("output_type", "embedding"),
            decay=float(cfg.get("decay", 0.9)),
            threshold=float(cfg.get("threshold", 1.0)),
        )
    elif model_name == "MLP":
        return MLP(
            input_dim=int(cfg.get("feature_dim", 12)),
            hidden_dims=cfg.get("hidden_dims", [128, 64]),
            output_dim=int(cfg.get("output_dim", 1)),
            dropout=float(cfg.get("dropout", 0.0)),
            activation=cfg.get("activation", "relu"),
            output_type=cfg.get("output_type", "embedding"),
        )
    elif model_name == "RBF":
        return RBF(
            input_dim=int(cfg.get("feature_dim", 12)),
            num_centers=int(cfg.get("hidden_dim", 100)),
            output_dim=int(cfg.get("output_dim", 1)),
            sigma=float(cfg.get("sigma", 1.0)),
            output_type=cfg.get("output_type", "embedding"),
        )
    elif model_name == "AE":
        return AutoEncoder(
            input_dim=int(cfg.get("feature_dim", 12)),
            hidden_dims=cfg.get("hidden_dims", [64]),
            latent_dim=int(cfg.get("hidden_dim", 32)),
            output_type=cfg.get("output_type", "embedding"),
        )
    elif model_name == "DAE":
        return DenoisingAE(
            input_dim=int(cfg.get("feature_dim", 12)),
            hidden_dims=cfg.get("hidden_dims", [64]),
            latent_dim=int(cfg.get("hidden_dim", 32)),
            noise_std=float(cfg.get("noise_std", 0.1)),
            output_type=cfg.get("output_type", "embedding"),
        )
    elif model_name == "SAE":
        return SparseAE(
            input_dim=int(cfg.get("feature_dim", 12)),
            hidden_dims=cfg.get("hidden_dims", [64]),
            latent_dim=int(cfg.get("hidden_dim", 32)),
            sparsity_target=float(cfg.get("sparsity_target", 0.05)),
            sparsity_weight=float(cfg.get("sparsity_weight", 0.1)),
            output_type=cfg.get("output_type", "embedding"),
        )
    elif model_name == "StackedAE":
        return StackedAutoEncoder(
            layer_sizes=[
                int(cfg.get("feature_dim", 12)),
                *cfg.get("hidden_dims", [64, 32]),
                int(cfg.get("latent_dim", 16)),
            ],
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "Hopfield":
        return HopfieldNetwork(
            size=int(cfg.get("feature_dim", 12)),
            output_type=cfg.get("output_type", "embedding"),
        )
    elif model_name == "RBM":
        return RBM(
            visible_dim=int(cfg.get("feature_dim", 12)),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            output_type=cfg.get("output_type", "embedding"),
        )
    elif model_name == "ESN":
        return EchoStateNetwork(
            input_dim=int(cfg.get("feature_dim", 12)),
            reservoir_dim=int(cfg.get("hidden_dim", 500)),
            output_dim=int(cfg.get("output_dim", 1)),
            spectral_radius=float(cfg.get("spectral_radius", 0.9)),
            sparsity=float(cfg.get("sparsity", 0.1)),
            output_type=cfg.get("output_type", "embedding"),
        )
    elif model_name == "ELM":
        return ELM(
            input_dim=int(cfg.get("feature_dim", 12)),
            hidden_dim=int(cfg.get("hidden_dim", 500)),
            output_dim=int(cfg.get("output_dim", 1)),
            activation=cfg.get("activation", "sigmoid"),
            output_type=cfg.get("output_type", "embedding"),
        )
    elif model_name == "SOM":
        return KohonenMap(
            input_dim=int(cfg.get("feature_dim", 12)),
            grid_size=cfg.get("grid_size", (10, 10)),
            output_type=cfg.get("output_type", "embedding"),
        )
    elif model_name == "Capsule":
        return CapsuleLayer(
            in_caps=int(cfg.get("in_caps", 8)),
            in_dim=int(cfg.get("in_dim", 16)),
            out_caps=int(cfg.get("out_caps", 4)),
            out_dim=int(cfg.get("out_dim", 32)),
            output_type=cfg.get("output_type", "embedding"),
        )
    elif model_name == "CNN":
        return RollingWindowCNN(
            input_dim=int(cfg.get("feature_dim", 12)),
            output_dim=1,
            seq_len=int(cfg.get("seq_len", 30)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            output_type=cfg.get("output_type", "embedding"),
        )
    elif model_name == "Perceptron":
        return Perceptron(
            input_dim=int(cfg.get("feature_dim", 12)),
            output_dim=int(cfg.get("output_dim", 1)),
            activation=cfg.get("activation", "sigmoid"),
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "MarkovChain":
        return MarkovChain(
            num_states=int(cfg.get("num_states", 10)),
            output_type=cfg.get("output_type", "prediction"),
            learnable=cfg.get("learnable", True),
        )
    elif model_name == "BM":
        return BoltzmannMachine(
            num_units=int(cfg.get("feature_dim", 12)),
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "DBN":
        return DeepBeliefNetwork(
            layer_sizes=[
                int(cfg.get("feature_dim", 12)),
                *cfg.get("hidden_dims", [64, 32]),
            ],
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "DCN":
        return DeepConvNet(
            input_dim=int(cfg.get("feature_dim", 12)),
            hidden_channels=cfg.get("hidden_channels", [32, 64, 128]),
            output_dim=int(cfg.get("output_dim", 1)),
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "Deconv":
        return DeconvNet(
            input_dim=int(cfg.get("feature_dim", 12)),
            hidden_channels=cfg.get("hidden_channels", [128, 64, 32]),
            output_dim=int(cfg.get("output_dim", 1)),
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "AutoDeconv":
        return AutoDeconvNet(
            input_dim=int(cfg.get("feature_dim", 12)),
            latent_dim=int(cfg.get("latent_dim", 64)),
            hidden_channels=cfg.get("hidden_channels", [32, 64, 128]),
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "DCIGN":
        latent_dim = int(cfg.get("latent_dim", 128))
        num_intrinsic = int(cfg.get("num_intrinsic", latent_dim // 4))
        num_extrinsic = latent_dim - num_intrinsic
        return DCIGN(
            input_dim=int(cfg.get("feature_dim", 12)),
            latent_dim=latent_dim,
            hidden_channels=cfg.get("hidden_channels", [32, 64, 128, 256]),
            num_intrinsic=num_intrinsic,
            num_extrinsic=num_extrinsic,
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "LSM":
        return LiquidStateMachine(
            input_dim=int(cfg.get("feature_dim", 12)),
            liquid_size=int(cfg.get("liquid_size", 200)),
            output_dim=int(cfg.get("output_dim", 1)),
            connection_prob=float(cfg.get("connection_prob", 0.3)),
            spectral_radius=float(cfg.get("spectral_radius", 1.2)),
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "ResNet":
        return DeepResNet(
            input_dim=int(cfg.get("feature_dim", 12)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            num_blocks=int(cfg.get("num_blocks", 4)),
            output_dim=int(cfg.get("output_dim", 1)),
            use_conv=cfg.get("use_conv", False),
            dropout=float(cfg.get("dropout", 0.1)),
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "DNC":
        return DNC(
            input_dim=int(cfg.get("feature_dim", 12)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            memory_size=int(cfg.get("memory_size", 64)),
            memory_dim=int(cfg.get("memory_dim", 32)),
            num_reads=int(cfg.get("num_reads", 4)),
            output_dim=int(cfg.get("output_dim", 1)),
            controller_type=cfg.get("controller_type", "lstm"),
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "NTM":
        return NTM(
            input_dim=int(cfg.get("feature_dim", 12)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            memory_size=int(cfg.get("memory_size", 128)),
            memory_dim=int(cfg.get("memory_dim", 20)),
            num_reads=int(cfg.get("num_reads", 1)),
            num_writes=int(cfg.get("num_writes", 1)),
            output_dim=int(cfg.get("output_dim", 1)),
            controller_type=cfg.get("controller_type", "lstm"),
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "Attention":
        return AttentionNetwork(
            input_dim=int(cfg.get("feature_dim", 12)),
            d_model=int(cfg.get("d_model", 128)),
            num_layers=int(cfg.get("num_layers", 4)),
            num_heads=int(cfg.get("num_heads", 8)),
            d_ff=int(cfg.get("d_ff", 512)),
            output_dim=int(cfg.get("output_dim", 1)),
            dropout=float(cfg.get("dropout", 0.1)),
            max_seq_len=int(cfg.get("max_seq_len", 1000)),
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "Flow":
        return NormalizingFlow(
            input_dim=int(cfg.get("feature_dim", 12)),
            num_layers=int(cfg.get("num_layers", 4)),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            seq_len=int(cfg.get("seq_len", 1)),
        )
    elif model_name == "NODE":
        return NeuralODE(
            input_dim=int(cfg.get("feature_dim", 12)),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            output_dim=int(cfg.get("output_dim", 1)),
            num_layers=int(cfg.get("num_layers", 2)),
            time_steps=int(cfg.get("seq_len", 10)),
            horizon=float(cfg.get("horizon", 1.0)),
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "LVQ":
        return LVQ(
            input_dim=int(cfg.get("feature_dim", 12)),
            num_classes=int(cfg.get("num_classes", 2)),
            prototypes_per_class=int(cfg.get("prototypes_per_class", 1)),
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "PINN":
        return PINN(
            input_dim=int(cfg.get("feature_dim", 2)),
            hidden_dim=int(cfg.get("hidden_dim", 20)),
            output_dim=int(cfg.get("output_dim", 1)),
            num_layers=int(cfg.get("num_layers", 4)),
            activation=cfg.get("activation", "tanh"),
            output_type=cfg.get("output_type", "prediction"),
        )
    elif model_name == "VAE":
        return VAE(
            input_dim=int(cfg.get("feature_dim", 12)),
            latent_dim=int(cfg.get("latent_dim", 16)),
            d_model=int(cfg.get("d_model", 128)),
            seq_len=int(cfg.get("seq_len", 100)),
            pred_len=int(cfg.get("pred_len", 20)),
            encoder_type=cfg.get("encoder_type", "transformer"),
            decoder_type=cfg.get("decoder_type", None),
            n_layers=int(cfg.get("num_layers", 3)),
            n_heads=int(cfg.get("num_heads", 8)),
            d_ff=int(cfg.get("d_ff", 512)),
            dropout=float(cfg.get("dropout", 0.1)),
            activation=cfg.get("activation", "gelu"),
        )

    return None
