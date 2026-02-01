from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Sample 2D tensor fixture."""
    return torch.randn(32, 64)

@pytest.fixture
def sequence_tensor() -> torch.Tensor:
    """Sample sequence tensor (B, T, F) fixture."""
    return torch.randn(16, 30, 8)

@pytest.fixture
def simple_tensordict() -> TensorDict:
    """Simple TensorDict fixture."""
    return TensorDict({
        "observation": torch.randn(8, 30, 6),
        "action": torch.zeros(8, dtype=torch.long),
        "reward": torch.zeros(8, 1),
    }, batch_size=[8])

@pytest.fixture
def obs_tensordict() -> TensorDict:
    """Observation-only TensorDict fixture."""
    return TensorDict({
        "observation": torch.randn(4, 30, 6),
    }, batch_size=[4])

__all__ = ["obs_tensordict", "sample_tensor", "sequence_tensor", "simple_tensordict"]
