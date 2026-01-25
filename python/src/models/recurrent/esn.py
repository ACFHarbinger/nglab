"""
Echo State Network (ESN) implementation.
"""

import torch
from torch import nn


class EchoStateNetwork(nn.Module):
    """
    Echo State Network (ESN).
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        reservoir_dim: int,
        output_dim: int,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        output_type: str = "prediction",
    ) -> None:
        """
        Initialize ESN.
        """
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.output_type = output_type

        self.register_buffer("win", (torch.rand(reservoir_dim, input_dim) - 0.5) * 2.0)
        self.win: torch.Tensor
        w_res = (torch.rand(reservoir_dim, reservoir_dim) - 0.5) * 2.0
        mask = torch.rand(reservoir_dim, reservoir_dim) < sparsity
        w_res = w_res * mask.float()

        eigenvalues = torch.linalg.eigvals(w_res)
        max_eigen = torch.max(torch.abs(eigenvalues))
        w_res = w_res * (spectral_radius / max_eigen)
        self.register_buffer("wres", w_res)
        self.wres: torch.Tensor

        self.readout = nn.Linear(reservoir_dim + input_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Calculate forward pass."""
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.reservoir_dim, device=x.device)

        states = []
        outputs = []
        for t in range(seq_len):
            u = x[:, t, :]
            h = torch.tanh(
                torch.matmul(u, self.win.t()) + torch.matmul(h, self.wres.t())
            )

            combined = torch.cat([u, h], dim=1)
            out = self.readout(combined)
            states.append(h)
            outputs.append(out)

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            res = torch.stack(states, dim=1)
        else:
            res = torch.stack(outputs, dim=1)

        if not return_sequence:
            return res[:, -1, :]
        return res
