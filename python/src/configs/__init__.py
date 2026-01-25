
from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from .algorithm import AlgorithmConfig, PPOConfig, SACConfig
from .base import BaseConfig
from .data import DataConfig, PolymarketConfig
from .env import EnvConfig, TradingEnvConfig
from .model import LSTMConfig, ModelConfig


@dataclass
class TrainConfig(BaseConfig):
    """Root configuration for training."""
    seed: int = 42
    task: str = "rl"
    device: str = "cpu"
    max_epochs: int = 10
    gradient_clip_val: float = 1.0
    val_check_interval: float = 1.0

    # Default to specific configs, can be overridden
    algorithm: AlgorithmConfig = field(default_factory=PPOConfig)
    env: EnvConfig = field(default_factory=TradingEnvConfig)
    model: ModelConfig = field(default_factory=LSTMConfig)
    data: DataConfig = field(default_factory=PolymarketConfig)


def register_configs() -> None:
    """Register structured configs with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    
    # Root config
    cs.store(name="config", node=TrainConfig)

    # Algorithm configs
    cs.store(group="algorithm", name="base", node=AlgorithmConfig)
    cs.store(group="algorithm", name="ppo", node=PPOConfig)
    cs.store(group="algorithm", name="sac", node=SACConfig)

    # Env configs
    cs.store(group="env", name="base", node=EnvConfig)
    cs.store(group="env", name="trading", node=TradingEnvConfig)

    # Model configs
    cs.store(group="model", name="base", node=ModelConfig)
    cs.store(group="model", name="lstm", node=LSTMConfig)

    # Data configs
    cs.store(group="data", name="base", node=DataConfig)
    cs.store(group="data", name="polymarket", node=PolymarketConfig)


# Explicit exports
Config = TrainConfig

__all__ = [
    "TrainConfig",
    "Config",
    "register_configs",
    "AlgorithmConfig",
    "PPOConfig",
    "SACConfig",
    "BaseConfig",
    "DataConfig",
    "PolymarketConfig",
    "EnvConfig",
    "TradingEnvConfig",
    "ModelConfig",
    "LSTMConfig",
]
