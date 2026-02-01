from __future__ import annotations

import pytest

from python.src.configs import (
    EnvConfig,
    ModelConfig,
    PPOConfig,
    TradingEnvConfig,
    TrainConfig,
)


@pytest.fixture
def base_config() -> TrainConfig:
    """Base training configuration fixture."""
    return TrainConfig(
        seed=42,
        task="rl",
        device="cpu",
        max_epochs=2
    )

@pytest.fixture
def model_config() -> ModelConfig:
    """Model configuration fixture."""
    return ModelConfig(
        name="test_model",
        hidden_dim=128,
        dropout=0.1
    )

@pytest.fixture
def env_config() -> EnvConfig:
    """Environment configuration fixture."""
    return TradingEnvConfig(
        lookback=30,
        max_steps=100
    )

@pytest.fixture
def ppo_config() -> PPOConfig:
    """PPO algorithm configuration fixture."""
    return PPOConfig(
        learning_rate=3e-4,
        n_steps=128,
        batch_size=32
    )

__all__ = ["base_config", "env_config", "model_config", "ppo_config"]
