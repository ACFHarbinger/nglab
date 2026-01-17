import pytest
import torch


@pytest.fixture
def mac_dummy_input():
    """Returns a dummy input tensor for classical models (Batch, Seq, Feat) = (4, 30, 10)."""
    return torch.randn(4, 30, 10)


@pytest.fixture
def classical_cfg():
    """Returns a default classical model configuration."""
    return {"name": "LinearRegression", "feature_dim": 10, "output_dim": 1}
