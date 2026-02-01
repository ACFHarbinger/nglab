from __future__ import annotations

import argparse
from typing import Any

import pytest


@pytest.fixture
def mock_args() -> argparse.Namespace:
    """Mock CLI arguments."""
    return argparse.Namespace(
        command="train",
        config="python/src/conf/config.yaml",
        overrides={"model.hidden_dim": "256"}
    )

@pytest.fixture
def train_args() -> dict[str, Any]:
    """Mock dictionary of training arguments."""
    return {
        "task": "rl",
        "device": "cpu",
        "max_epochs": 5,
        "seed": 42
    }

__all__ = ["mock_args", "train_args"]
