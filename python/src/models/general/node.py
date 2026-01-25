"""
Neural Ordinary Differential Equations (Neural ODE).

This module implements a Neural ODE model, where the derivative of the hidden state
is parameterized by a neural network: dh/dt = f(h, t).
"""

from collections.abc import Callable
from typing import cast

import torch
from torch import nn


def odesolve(
    func: Callable[[float | torch.Tensor, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t: torch.Tensor,
    method: str = "rk4",
) -> torch.Tensor:
    """
    Solve an initial value problem (IVP) using a fixed-step solver.

    Args:
        func: Function f(t, y) returning dy/dt.
        y0: Initial state (batch_size, state_dim).
        t: Time points to evaluate at (num_steps,). Must be increasing.
        method: Solver method ('rk4' or 'euler').

    Returns:
        y: Solution trajectory (num_steps, batch_size, state_dim).
    """
    device = y0.device
    num_steps = len(t)
    y = torch.zeros((num_steps, *y0.shape), device=device)
    y[0] = y0

    curr_y = y0

    for i in range(num_steps - 1):
        t0 = t[i]
        t1 = t[i + 1]
        dt = t1 - t0

        if method == "euler":
            dy = func(t0, curr_y)
            curr_y = curr_y + dy * dt

        elif method == "rk4":
            k1 = func(t0, curr_y)
            k2 = func(t0 + dt / 2, curr_y + (dt / 2) * k1)
            k3 = func(t0 + dt / 2, curr_y + (dt / 2) * k2)
            k4 = func(t0 + dt, curr_y + dt * k3)

            curr_y = curr_y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        else:
            raise ValueError(f"Unknown method: {method}")

        y[i + 1] = curr_y

    return y


class ODEFunc(nn.Module):
    """
    Wrapper for the derivative function to handle time conditioning easily.
    """

    def __init__(self, net: nn.Module) -> None:
        """Initialize with a neural network."""
        super().__init__()
        self.net = net

    def forward(self, t: float | torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the derivative dy/dt."""
        # For this implementation, we assume simple autonomous dynamics f(y)
        return cast(torch.Tensor, self.net(y))


class NeuralODE(nn.Module):
    """
    Neural Ordinary Differential Equation Model.

    The model predicts the evolution of a state 'z' over time:
    dz/dt = f(z, t, theta)
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,  # Projection
        num_layers: int = 2,
        time_steps: int = 10,
        horizon: float = 1.0,
        output_type: str = "prediction",  # 'prediction' or 'embedding'
    ) -> None:
        """
        Initialize the Neural ODE.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_steps = time_steps
        self.horizon = horizon
        self.output_type = output_type

        # Derivative network f(z)
        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())  # Tanh is often good for ODEs (smooth)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(
            nn.Linear(hidden_dim, input_dim)
        )  # Output must act on state space

        self.diffeq = ODEFunc(nn.Sequential(*layers))

        # Final projection if output_dim != input_dim
        self.projection: nn.Module
        if output_dim != input_dim:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Initial state (batch_size, input_dim).
            t: Time points (num_steps,). Defaults to linspace(0, horizon, time_steps).

        Returns:
            trajectory or prediction.
        """
        if t is None:
            t = torch.linspace(0, self.horizon, self.time_steps, device=x.device)

        # Solve ODE
        # Shape: (num_steps, batch, input_dim)
        traj = odesolve(self.diffeq, x, t, method="rk4")

        # Permute to (batch, num_steps, ...)
        # We want to swap the first two dimensions
        dims = list(range(traj.dim()))
        dims[0], dims[1] = dims[1], dims[0]
        traj = traj.permute(*dims)

        if self.output_type == "embedding":
            return traj  # Return full trajectory for embedding

        # For prediction, we might want the last state or projected trajectory
        # Project each step
        out = self.projection(traj)  # (batch, steps, output_dim)

        return cast(torch.Tensor, out)
