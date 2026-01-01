"""
SAC Training Script for Trading Environment

This script trains a Soft Actor-Critic (SAC) agent
on the Rust-backed TradingEnv using Stable-Baselines3.

NOTE: SAC requires a continuous action space. TradingEnv uses Discrete(3).
This script uses a custom wrapper to convert the discrete space to continuous.
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import TradingEnv


class ContinuousActionWrapper(gym.ActionWrapper):
    """
    Wrapper to convert discrete action space to continuous.
    Maps a continuous action in [-1, 1] to discrete {0, 1, 2}.
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Discrete), "Expected Discrete action space"
        self.n_actions = env.action_space.n
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    def action(self, action):
        # Map continuous [-1, 1] to discrete action
        # -1 to -0.33: Sell (2)
        # -0.33 to 0.33: Hold (0)
        # 0.33 to 1: Buy (1)
        continuous_action = np.clip(action[0], -1.0, 1.0)
        if continuous_action < -0.33:
            return 2  # Sell
        elif continuous_action > 0.33:
            return 1  # Buy
        else:
            return 0  # Hold


def make_env(rank: int, seed: int = 0, lookback: int = 30, max_steps: int = 1000):
    """Create a wrapped TradingEnv instance for SAC."""
    def _init():
        env = TradingEnv(lookback=lookback, max_steps=max_steps)
        env = ContinuousActionWrapper(env)
        env = Monitor(env)
        return env
    return _init


def train_sac(args):
    """Train a SAC agent on TradingEnv."""
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create vectorized environment (SAC doesn't benefit much from parallel envs)
    env = DummyVecEnv([make_env(0, args.seed, args.lookback, args.max_steps)])
    
    # Evaluation environment
    eval_env = DummyVecEnv([make_env(0, args.seed + 100, args.lookback, args.max_steps)])
    
    # Define the SAC model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        tensorboard_log=log_dir,
        seed=args.seed,
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.save_dir,
        name_prefix="sac_trading"
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False
    )
    
    # Train
    print(f"Starting SAC training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    final_path = os.path.join(args.save_dir, "sac_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on TradingEnv")
    parser.add_argument("--total_timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--save_dir", type=str, default="models/sac_trading", help="Directory to save models")
    parser.add_argument("--lookback", type=int, default=30, help="Lookback window for observations")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Replay buffer size")
    parser.add_argument("--learning_starts", type=int, default=1000, help="Steps before learning starts")
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--train_freq", type=int, default=1, help="Training frequency")
    parser.add_argument("--gradient_steps", type=int, default=1, help="Gradient steps per update")
    parser.add_argument("--ent_coef", type=str, default="auto", help="Entropy coefficient (auto or float)")
    parser.add_argument("--checkpoint_freq", type=int, default=10_000, help="Checkpoint save frequency")
    parser.add_argument("--eval_freq", type=int, default=5_000, help="Evaluation frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    train_sac(args)
