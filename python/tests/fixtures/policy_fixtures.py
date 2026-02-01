from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from torch import nn

from python.src.policies.neural import NeuralPolicy


@pytest.fixture
def mock_neural_model():
    """Fixture for a mock neural model that satisfies the ModelProtocol."""
    model = MagicMock(spec=nn.Module)
    # Set up some default behavior if needed
    return model


@pytest.fixture
def neural_policy(mock_neural_model):
    """Fixture for a NeuralPolicy instance."""
    cfg = {"device": "cpu"}
    return NeuralPolicy(model=mock_neural_model, cfg=cfg)


@pytest.fixture
def policy_config():
    """Generic policy configuration fixture."""
    return {
        "name": "neural",
        "device": "cpu",
        "backbone_kwargs": {
            "hidden_dim": 128
        }
    }
