"""
SAC Training Script for Trading Environment

This script trains a Soft Actor-Critic (SAC) agent
on the Rust-backed TradingEnv using Stable-Baselines3.

NOTE: SAC requires a continuous action space. TradingEnv uses Discrete(3).
This script uses a custom wrapper to convert the discrete space to continuous.
"""

import os
from collections.abc import Callable
from typing import Any, cast

import gymnasium as gym
import hydra
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from python.src.configs import register_configs
from python.src.envs import TradingEnv

# Register structured configs
register_configs()


class ContinuousActionWrapper(gym.ActionWrapper[Any, Any, Any]):
    """
    Wrapper to convert discrete action space to continuous.
    Maps a continuous action in [-1, 1] to discrete {0, 1, 2}.
    """

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        """
        Initialize the wrapper.
        """
        super().__init__(env)
        action_space = env.action_space
        assert isinstance(
            action_space, spaces.Discrete
        ), "Expected Discrete action space"
        self.n_actions = int(cast(Any, action_space).n)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def action(self, action: np.ndarray[Any, Any]) -> int:
        """
        Convert continuous action to discrete.
        """
        # Map continuous [-1, 1] to discrete action
        # -1 to -0.33: Sell (2)
        # -0.33 to 0.33: Hold (0)
        # 0.33 to 1: Buy (1)
        continuous_action = float(np.clip(action[0], -1.0, 1.0))
        if continuous_action < -0.33:
            return 2  # Sell
        elif continuous_action > 0.33:
            return 1  # Buy
        else:
            return 0  # Hold


def make_env(
    rank: int, seed: int = 0, lookback: int = 30, max_steps: int = 1000
) -> Callable[[], gym.Env[Any, Any]]:
    """Create a wrapped TradingEnv instance for SAC."""

    def _init() -> gym.Env[Any, Any]:
        env: gym.Env[Any, Any] = TradingEnv(lookback=lookback, max_steps=max_steps)
        env = ContinuousActionWrapper(env)
        env = Monitor(env)
        return env

    return _init


@hydra.main(version_base=None, config_path=None, config_name="config")
def train_sac(cfg: DictConfig) -> None:
    """Train a SAC agent on TradingEnv."""

    # Create output directory
    os.makedirs(cfg.algorithm.save_dir, exist_ok=True)
    log_dir = os.path.join(cfg.algorithm.save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create vectorized environment (SAC doesn't benefit much from parallel envs)
    env = DummyVecEnv([make_env(0, cfg.seed, cfg.env.lookback, cfg.env.max_steps)])

    # Evaluation environment
    eval_env = DummyVecEnv(
        [make_env(0, cfg.seed + 100, cfg.env.lookback, cfg.env.max_steps)]
    )

    # Define the SAC model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=cfg.algorithm.learning_rate,
        buffer_size=cfg.algorithm.buffer_size,
        learning_starts=cfg.algorithm.learning_starts,
        batch_size=cfg.algorithm.batch_size,
        tau=cfg.algorithm.tau,
        gamma=cfg.algorithm.gamma,
        train_freq=cfg.algorithm.train_freq,
        gradient_steps=cfg.algorithm.gradient_steps,
        ent_coef=cfg.algorithm.ent_coef,
        tensorboard_log=log_dir,
        seed=cfg.seed,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.algorithm.checkpoint_freq,
        save_path=cfg.algorithm.save_dir,
        name_prefix="sac_trading",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=cfg.algorithm.save_dir,
        log_path=log_dir,
        eval_freq=cfg.algorithm.eval_freq,
        deterministic=True,
        render=False,
    )

    # Train
    print(f"Starting SAC training for {cfg.algorithm.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=cfg.algorithm.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(cfg.algorithm.save_dir, "sac_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    train_sac()
