"""
Agent Evaluation Script

This script evaluates trained PPO or SAC agents on the TradingEnv
and computes performance metrics like Sharpe Ratio and Max Drawdown.
"""

import argparse
from typing import Any

import numpy as np
from stable_baselines3 import PPO, SAC

from pipeline.core.train_sac import ContinuousActionWrapper
from python.src.envs import TradingEnv


def calculate_metrics(portfolio_values: list[float]) -> dict[str, Any]:
    """Calculate trading performance metrics."""
    portfolio_arr = np.array(portfolio_values)

    # Returns
    returns = np.diff(portfolio_arr) / portfolio_arr[:-1]

    # Sharpe Ratio (annualized, assuming 252 trading days)
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1) if len(returns) > 1 else 1e-8
    sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0.0

    # Max Drawdown
    cumulative_max = np.maximum.accumulate(portfolio_arr)
    drawdowns = (cumulative_max - portfolio_arr) / cumulative_max
    max_drawdown = np.max(drawdowns)

    # Total Return
    total_return = (portfolio_arr[-1] - portfolio_arr[0]) / portfolio_arr[0]

    return {
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "total_return": total_return,
        "final_value": portfolio_arr[-1],
        "mean_return": mean_return,
        "std_return": std_return,
    }


def evaluate_agent(
    model_path: str,
    agent_type: str,
    n_episodes: int = 10,
    lookback: int = 30,
    max_steps: int = 1000,
) -> dict[str, Any]:
    """Evaluate a trained agent over multiple episodes."""

    # Load model
    model: PPO | SAC
    if agent_type.lower() == "ppo":
        model = PPO.load(model_path)
        env = TradingEnv(lookback=lookback, max_steps=max_steps)
    elif agent_type.lower() == "sac":
        model = SAC.load(model_path)
        env = ContinuousActionWrapper(
            TradingEnv(lookback=lookback, max_steps=max_steps)
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    all_portfolio_values = []
    total_rewards = []
    episode_lengths = []

    for _episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward: float = 0.0
        steps = 0
        portfolio_values = [10000.0]  # Initial capital

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                # Use a specific variable for the step action to avoid type mismatch
                step_action: Any = int(action.item())
            else:
                step_action = action
            obs, reward, terminated, truncated, info = env.step(step_action)
            done = terminated or truncated
            episode_reward += float(reward)
            steps += 1
            portfolio_values.append(info.get("portfolio_value", portfolio_values[-1]))

        all_portfolio_values.append(portfolio_values)
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)

    env.close()

    # Aggregate metrics across episodes
    aggregated_metrics = []
    for pv in all_portfolio_values:
        metrics = calculate_metrics(pv)
        aggregated_metrics.append(metrics)

    # Average metrics
    avg_metrics = {
        "avg_sharpe_ratio": np.mean([m["sharpe_ratio"] for m in aggregated_metrics]),
        "avg_max_drawdown": np.mean([m["max_drawdown"] for m in aggregated_metrics]),
        "avg_total_return": np.mean([m["total_return"] for m in aggregated_metrics]),
        "avg_final_value": np.mean([m["final_value"] for m in aggregated_metrics]),
        "avg_episode_reward": np.mean(total_rewards),
        "avg_episode_length": np.mean(episode_lengths),
    }

    return avg_metrics


def main(args: argparse.Namespace) -> None:
    """
    Main evaluation entry point.
    """
    print(f"Evaluating {args.agent_type.upper()} agent from: {args.model_path}")
    print(f"Running {args.n_episodes} evaluation episodes...\n")

    metrics = evaluate_agent(
        args.model_path, args.agent_type, args.n_episodes, args.lookback, args.max_steps
    )

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Average Sharpe Ratio:    {metrics['avg_sharpe_ratio']:.4f}")
    print(f"Average Max Drawdown:    {metrics['avg_max_drawdown']:.4%}")
    print(f"Average Total Return:    {metrics['avg_total_return']:.4%}")
    print(f"Average Final Value:     ${metrics['avg_final_value']:.2f}")
    print(f"Average Episode Reward:  {metrics['avg_episode_reward']:.2f}")
    print(f"Average Episode Length:  {metrics['avg_episode_length']:.0f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        required=True,
        choices=["ppo", "sac"],
        help="Agent type",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--lookback", type=int, default=30, help="Lookback window for observations"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Max steps per episode"
    )

    args = parser.parse_args()
    main(args)
