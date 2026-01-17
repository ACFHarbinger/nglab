import pytest
import torch


@pytest.fixture
def deep_dummy_input():
    """Returns a dummy input tensor for deep models (Batch, Seq, Feat) = (4, 30, 10)."""
    return torch.randn(4, 30, 10)


@pytest.fixture
def vae_config():
    """Config specifically for VAE."""
    return {
        "input_dim": 10,
        "latent_dim": 16,
        "d_model": 32,
        "seq_len": 30,
        "pred_len": 5,
        "encoder_type": "lstm",
    }


@pytest.fixture
def deep_model_config():
    """General config for deep models."""
    return {"input_dim": 10, "hidden_dim": 64, "output_dim": 1, "n_layers": 2}
