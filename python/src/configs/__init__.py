from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .algorithm import AlgorithmConfig, PPOConfig, SACConfig
from .data import DataConfig, PolymarketConfig
from .env import EnvConfig, TradingEnvConfig
from .model import LSTMConfig, ModelConfig


@dataclass
class TrainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            {"algorithm": "ppo"},
            {"env": "trading"},
            {"model": "lstm"},
            {"data": "polymarket"},
            "_self_",
        ]
    )

    seed: int = 42
    task: str = "rl"
    device: str = "cpu"
    max_epochs: int = 10
    gradient_clip_val: float = 1.0
    val_check_interval: float = 1.0

    algorithm: AlgorithmConfig = MISSING
    env: EnvConfig = MISSING
    model: ModelConfig = MISSING
    data: DataConfig = MISSING


def register_configs() -> None:
    cs = ConfigStore.instance()

    # Register the main config schema
    cs.store(name="config", node=TrainConfig)

    # Register algorithm configs
    cs.store(group="algorithm", name="ppo", node=PPOConfig)
    cs.store(group="algorithm", name="sac", node=SACConfig)

    # Register env configs
    cs.store(group="env", name="trading", node=TradingEnvConfig)

    # Register model configs
    cs.store(group="model", name="lstm", node=LSTMConfig)

    # Register data configs
    cs.store(group="data", name="polymarket", node=PolymarketConfig)
