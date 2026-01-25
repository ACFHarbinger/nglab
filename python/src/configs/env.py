
from __future__ import annotations

from dataclasses import dataclass

from python.src.configs.base import BaseConfig


@dataclass
class EnvConfig(BaseConfig):
    lookback: int = 30
    max_steps: int = 1000
    num_envs: int = 1
    device: str = "cpu"


@dataclass
class TradingEnvConfig(EnvConfig):
    pass
