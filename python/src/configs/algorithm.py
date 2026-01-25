from __future__ import annotations

from dataclasses import dataclass

from python.src.configs.base import BaseConfig

__all__ = ["AlgorithmConfig", "PPOConfig", "SACConfig"]


@dataclass
class AlgorithmConfig(BaseConfig):
    name: str = "base"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    save_dir: str = "models/base"


@dataclass
class PPOConfig(AlgorithmConfig):
    name: str = "ppo"
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    n_steps: int = 2048
    n_epochs: int = 10
    batch_size: int = 64
    clip_epsilon: float = 0.2
    frames_per_batch: int = 1000
    total_frames: int = 1_000_000
    ppo_epochs: int = 10
    mini_batch_size: int = 64
    max_grad_norm: float = 1.0
    log_step: int = 1
    checkpoint_freq: int = 10000
    eval_freq: int = 5000
    total_timesteps: int = 100_000
    save_dir: str = "models/ppo_trading"
    run_name: str = "ppo_run"
    no_tensorboard: bool = False
    no_progress_bar: bool = False


@dataclass
class SACConfig(AlgorithmConfig):
    name: str = "sac"
    tau: float = 0.005
    buffer_size: int = 100_000
    learning_starts: int = 1000
    batch_size: int = 256
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str | float = "auto"
    checkpoint_freq: int = 10000
    eval_freq: int = 5000
    total_timesteps: int = 100_000
    save_dir: str = "models/sac_trading"
