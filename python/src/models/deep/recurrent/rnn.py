"""
Recurrent Neural Network (RNN) implementations (LSTM, GRU).
"""


import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from typing import cast


class LSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) network for sequence processing.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        output_dim: int,
        dropout: float = 0.0,
        output_type: str = "prediction",
        apply_softmax: bool = False,
    ) -> None:
        """
        Initialize the LSTM.
        """
        super().__init__()
        self.output_type = output_type
        self.apply_softmax = apply_softmax
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, return_embedding: bool | None = None, return_sequence: bool = False) -> torch.Tensor:
        """
        Forward pass.
        """
        # h0, c0 initialized to zeros by default if not provided
        out, (_h_n, _c_n) = self.lstm(x)

        # Determine if we want full sequence or last step
        if return_sequence:
            state = out  # (B, L, H)
        else:
            state = out[:, -1, :]  # (B, H)

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            return cast(torch.Tensor, state)

        output = self.fc(state)

        if self.apply_softmax and self.output_type == "prediction":
            output = F.softmax(output, dim=-1)

        return cast(torch.Tensor, output)


class GRU(nn.Module):
    """
    Gated Recurrent Unit (GRU) network for sequence processing.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        output_dim: int,
        dropout: float = 0.0,
        output_type: str = "prediction",
    ) -> None:
        """
        Initialize the GRU.
        """
        super().__init__()
        self.output_type = output_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, return_embedding: bool | None = None, return_sequence: bool = False) -> torch.Tensor:
        """
        Forward pass.
        """
        # h0 initialized to zeros by default if not provided
        out, _h_n = self.gru(x)

        # Determine if we want full sequence or last step
        if return_sequence:
            state = out
        else:
            state = out[:, -1, :]

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            return cast(torch.Tensor, state)

        return cast(torch.Tensor, self.fc(state))
