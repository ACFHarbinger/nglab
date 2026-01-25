from dataclasses import dataclass


@dataclass
class EnvConfig:
    lookback: int = 30
    max_steps: int = 1000
    num_envs: int = 1
    device: str = (
        "cpu"  # Will be interpolated in main config usually, but defaults here
    )


@dataclass
class TradingEnvConfig(EnvConfig):
    pass
