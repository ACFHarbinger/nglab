"""
Spiking Neural Network (SNN) implementation using Surrogate Gradients.
"""

from typing import Any

import torch
from torch import nn


class SurrogateHeaviside(torch.autograd.Function):
    """
    Heaviside step function with a surrogate gradient for backpropagation.
    We use the Fast Sigmoid derivative as the surrogate gradient.
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, alpha: float = 25.0) -> torch.Tensor:
        """Forward pass."""
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input > 0).float()

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:
        """Backward pass with surrogate gradient."""
        (input_tensor,) = ctx.saved_tensors
        # Surrogate gradient: alpha / (1 + |alpha * input|)^2
        # This is the derivative of the fast sigmoid function.
        grad_input = grad_outputs * (
            ctx.alpha / (1 + torch.abs(ctx.alpha * input_tensor)).pow(2)
        )
        return grad_input, None


def surrogate_heaviside(x: torch.Tensor, alpha: float = 25.0) -> torch.Tensor:
    """Compute Heaviside with surrogate gradient."""
    return SurrogateHeaviside.apply(x, alpha)  # type: ignore


class LIFCell(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Neuron Cell.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        decay: float = 0.9,
        threshold: float = 1.0,
        alpha: float = 25.0,
    ) -> None:
        """Initialize LIF Cell."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.decay = decay
        self.threshold = threshold
        self.alpha = alpha

        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Returns:
            s_next, (u_next, s_next)
        """
        batch_size = x.size(0)

        if state is None:
            u = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            s = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            u, s = state

        # Linear transform of input
        i_inj = self.linear(x)

        # Update membrane potential
        u_next = self.decay * (u - s * self.threshold) + i_inj

        # Fire spike
        s_next = surrogate_heaviside(u_next - self.threshold, self.alpha)

        return s_next, (u_next, s_next)


class SNN(nn.Module):
    """
    Multi-layer Spiking Neural Network.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = 1,
        output_dim: int | None = None,
        decay: float = 0.9,
        threshold: float = 1.0,
        alpha: float = 25.0,
        dropout: float = 0.0,
        output_type: str = "prediction",
    ) -> None:
        """Initialize SNN."""
        super().__init__()
        self.output_type = output_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lif_layers = nn.ModuleList(
            [
                LIFCell(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    decay=decay,
                    threshold=threshold,
                    alpha=alpha,
                )
                for i in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim, output_dim) if output_dim else None

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass."""
        _batch_size, seq_len, _ = x.size()
        current_input = x

        for layer_idx, lif in enumerate(self.lif_layers):
            layer_outputs = []
            state = None
            for t in range(seq_len):
                input_t = current_input[:, t, :]
                spikes, state = lif(input_t, state)
                layer_outputs.append(spikes)

            current_input = torch.stack(layer_outputs, dim=1)

            if layer_idx < self.n_layers - 1:
                current_input = self.dropout(current_input)

        final_spikes = current_input

        if return_sequence:
            out = final_spikes
        else:
            out = final_spikes[:, -1, :]

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            return torch.as_tensor(out)

        if self.fc is not None:
            return torch.as_tensor(self.fc(out))
        else:
            return torch.as_tensor(out)
