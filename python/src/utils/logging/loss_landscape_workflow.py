"""
Workflow for generating and visualizing reward landscapes.
"""

from typing import Any, cast

import loss_landscapes
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from python.src.envs.trading_env import TradingEnv


def evaluate_model(model: Any, env: TradingEnv, num_steps: int = 100) -> float:
    """
    Evaluation function required by loss-landscapes.
    Returns the negative total reward (to treat it as a 'loss' to be minimized).
    """
    # loss-landscapes 3.0+ wraps the model in a wrapper that mimics nn.Module
    # but is not directly callable. We use .forward() explicitly.
    model.eval()
    obs, _ = env.reset()
    total_reward = 0.0

    # Try to get device from parameters
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")

    with torch.no_grad():
        for _ in range(num_steps):
            # Convert observation to tensor
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            # Invoke forward explicitly as the wrapper might not be callable
            action_logits = model.forward(obs_tensor)
            action_val = int(torch.argmax(action_logits, dim=-1).item())

            # TradingEnv expectations: action can be int or ndarray.
            # If step expects ndarray, we should provide it.
            obs, reward, terminated, truncated, _info = env.step(np.array([action_val]))
            total_reward += float(reward)

            if terminated or truncated:
                break

    return -total_reward


class SimplePolicy(nn.Module):
    """
    Simple 2-layer MLP policy for landscape analysis.
    """

    def __init__(
        self, input_dim: int = 12, hidden_dim: int = 64, output_dim: int = 1
    ) -> None:
        """
        Initialize.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        return cast(torch.Tensor, self.net(x))


def run_landscape_analysis() -> None:
    """
    Main function to run the landscape analysis and save a plot.
    """
    print("Initializing environment...")
    feature_dim = 12
    env = TradingEnv(lookback=30, max_steps=100, feature_dim=feature_dim)

    print("Initializing model...")
    model = SimplePolicy(input_dim=feature_dim)

    # We want to see how the landscape looks around the current weights
    # We'll use a random direction approach (standard for loss-landscapes)
    x_bound = 1.0
    y_bound = 1.0
    steps = 20

    print(f"Generating landscape surface ({steps}x{steps} grid)...")
    # Generate the landscape
    # deepcopy_model=True ensures the original model isn't modified
    landscape = loss_landscapes.random_plane(
        model,
        lambda m: evaluate_model(m, env),
        distance=1,
        steps=steps,
        normalization="filter",
        deepcopy_model=True,
    )

    print("Plotting landscape...")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    X = np.linspace(-x_bound, x_bound, steps)
    Y = np.linspace(-y_bound, y_bound, steps)
    X, Y = np.meshgrid(X, Y)
    Z = landscape

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    ax.set_title("3D Reward Landscape (Negative Total Reward)")
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    ax.set_zlabel("Loss (-Reward)")

    output_path = "loss_landscape_3d.png"
    plt.savefig(output_path)
    print(f"Landscape saved to {output_path}")


if __name__ == "__main__":
    run_landscape_analysis()
