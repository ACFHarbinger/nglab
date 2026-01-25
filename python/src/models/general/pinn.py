"""
Physics-Informed Neural Network (PINN).

This module implements a PINN architecture, which is essentially an MLP
that is designed to compute gradients with respect to its inputs (space/time)
to solve partial differential equations (PDEs).
"""

from typing import cast

import torch
from torch import nn


class PINN(nn.Module):
    """
    Physics-Informed Neural Network.

    A fully connected network that approximates a solution u(x, t) to a PDE.
    Includes methods to easily compute derivatives du/dx, du/dt, d^2u/dx^2, etc.

    Args:
        input_dim: Dimension of domain (e.g., 2 for x and t).
        hidden_dim: Number of neurons per layer.
        output_dim: Dimension of solution u (e.g., 1 for scalar field).
        num_layers: Number of hidden layers.
        activation: Activation function (tanh is standard for PINNs due to smoothness).
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        hidden_dim: int = 20,
        output_dim: int = 1,
        num_layers: int = 4,
        activation: str = "tanh",
        output_type: str = "prediction",
    ) -> None:
        """
        Initialize PINN.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_type = output_type

        # Activation
        act_fn: nn.Module
        if activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "sigmoid":
            act_fn = nn.Sigmoid()
        elif activation == "relu":
            act_fn = nn.ReLU()  # Not recommended for high-order derivatives
        elif activation == "gelu":
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act_fn)

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn)

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Collocation points (batch_size, input_dim).
               Requires requires_grad=True if computing physics loss outside.
        """
        return cast(torch.Tensor, self.net(x))

    def gradient(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient du/dx.

        Args:
            u: Output tensor (batch_size, output_dim).
            x: Input tensor (batch_size, input_dim).

        Returns:
            grad: Gradient tensor (batch_size, input_dim).
        """
        # We need grad_outputs=ones because output is (batch, output_dim)
        # For simplicity assuming output_dim=1 or we sum, but technically we want Jacobian.
        # Here assuming scalar u for typical PINN use case or element-wise.

        grads = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grads

    def laplacian(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian (sum of second derivatives).
        Note: This is expensive (O(d^2) or O(d) depending on implementation).
        For input_dim=1 (time), it is just d^2u/dt^2.
        """
        grad = self.gradient(u, x)
        lap = torch.zeros(x.shape[0], 1, device=x.device)
        for i in range(x.shape[1]):
            grad_i = grad[:, i : i + 1]  # Keep dim
            grad_outputs = torch.ones_like(grad_i)
            grad_2 = torch.autograd.grad(
                grad_i,
                x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            lap += grad_2[:, i : i + 1]
        return lap


def pinn_loss(
    u_pred: torch.Tensor,
    u_target: torch.Tensor | None,
    f_pred: torch.Tensor,  # Physics residual (e.g., u_t + u*u_x - u_xx)
    data_weight: float = 1.0,
    physics_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Compute PINN loss = MSE_data + MSE_physics.

    Args:
        u_pred: Predicted solution at data points.
        u_target: Target solution (boundary/initial conditions or observed data).
                  Can be None if solving purely unsupervised (unlikely).
        f_pred: Physics residual at collocation points. Should be close to 0.
    """
    mse_f = torch.mean(f_pred**2)

    loss = physics_weight * mse_f
    loss_dict = {"physics_loss": mse_f}

    if u_target is not None:
        mse_u = nn.functional.mse_loss(u_pred, u_target)
        loss += data_weight * mse_u
        loss_dict["data_loss"] = mse_u

    loss_dict["total_loss"] = loss
    return loss_dict
