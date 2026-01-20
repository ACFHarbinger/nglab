"""
PPO Training Script for Trading Environment

This script trains a Proximal Policy Optimization (PPO) agent
on the Rust-backed TradingEnv using Stable-Baselines3.
"""

import argparse
import os

from environment import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_env(rank: int, seed: int = 0, lookback: int = 30, max_steps: int = 1000):
    """Create a TradingEnv instance."""

    def _init():
        env = TradingEnv(lookback=lookback, max_steps=max_steps)
        env = Monitor(env)
        return env

    return _init


def train_ppo(args):
    """Train a PPO agent on TradingEnv."""

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create vectorized environment
    if args.num_envs > 1:
        env = SubprocVecEnv(
            [
                make_env(i, args.seed, args.lookback, args.max_steps)
                for i in range(args.num_envs)
            ]
        )
    else:
        env = DummyVecEnv([make_env(0, args.seed, args.lookback, args.max_steps)])

    # Evaluation environment
    eval_env = DummyVecEnv(
        [make_env(0, args.seed + 100, args.lookback, args.max_steps)]
    )

    # Define the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        tensorboard_log=log_dir,
        seed=args.seed,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.save_dir,
        name_prefix="ppo_trading",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )

    # Train
    print(f"Starting PPO training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(args.save_dir, "ppo_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on TradingEnv")
    parser.add_argument(
        "--total_timesteps", type=int, default=100_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models/ppo_trading",
        help="Directory to save models",
    )
    parser.add_argument(
        "--num_envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument(
        "--lookback", type=int, default=30, help="Lookback window for observations"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Max steps per episode"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument("--n_steps", type=int, default=2048, help="Steps per update")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument(
        "--n_epochs", type=int, default=10, help="Number of epochs per update"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument(
        "--clip_range", type=float, default=0.2, help="PPO clipping range"
    )
    parser.add_argument(
        "--ent_coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--checkpoint_freq", type=int, default=10_000, help="Checkpoint save frequency"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=5_000, help="Evaluation frequency"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train_ppo(args)
