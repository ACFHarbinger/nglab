"""
PPO Training Script for Trading Environment

This script trains a Proximal Policy Optimization (PPO) agent
on the Rust-backed TradingEnv using Stable-Baselines3.
"""

import os
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from python.src.configs import register_configs
from python.src.envs import TradingEnv

# Register structured configs
register_configs()


def make_env(
    rank: int, seed: int = 0, lookback: int = 30, max_steps: int = 1000
) -> Callable[[], gym.Env[Any, Any]]:
    """Create a TradingEnv instance."""

    def _init() -> gym.Env[Any, Any]:
        env: gym.Env[Any, Any] = TradingEnv(lookback=lookback, max_steps=max_steps)
        env = Monitor(env)
        return env

    return _init


@hydra.main(version_base=None, config_path=None, config_name="config")
def train_ppo(cfg: DictConfig) -> None:
    """Train a PPO agent on TradingEnv."""

    # Create output directory
    os.makedirs(cfg.algorithm.save_dir, exist_ok=True)
    log_dir = os.path.join(cfg.algorithm.save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create vectorized environment
    env: Any
    if cfg.env.num_envs > 1:
        env = SubprocVecEnv(
            [
                make_env(i, cfg.seed, cfg.env.lookback, cfg.env.max_steps)
                for i in range(cfg.env.num_envs)
            ]
        )
    else:
        env = DummyVecEnv([make_env(0, cfg.seed, cfg.env.lookback, cfg.env.max_steps)])

    # Evaluation environment
    eval_env = DummyVecEnv(
        [make_env(0, cfg.seed + 100, cfg.env.lookback, cfg.env.max_steps)]
    )

    # Define the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=cfg.algorithm.learning_rate,
        n_steps=cfg.algorithm.n_steps,
        batch_size=cfg.algorithm.batch_size,
        n_epochs=cfg.algorithm.n_epochs,
        gamma=cfg.algorithm.gamma,
        gae_lambda=cfg.algorithm.gae_lambda,
        clip_range=cfg.algorithm.clip_range,
        ent_coef=cfg.algorithm.ent_coef,
        tensorboard_log=log_dir,
        seed=cfg.seed,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.algorithm.checkpoint_freq,
        save_path=cfg.algorithm.save_dir,
        name_prefix="ppo_trading",
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
    print(f"Starting PPO training for {cfg.algorithm.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=cfg.algorithm.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=not cfg.algorithm.no_progress_bar,
    )

    # Save final model
    final_path = os.path.join(cfg.algorithm.save_dir, "ppo_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    train_ppo()
