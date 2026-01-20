"""
Recurrent Neural Network (RNN) implementations (LSTM, GRU).
"""

from torch import nn


class LSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) network for sequence processing.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_layers,
        output_dim,
        dropout=0.0,
        output_type="prediction",
        apply_softmax=False,
    ):
        """
        Initialize the LSTM.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden state dimension.
            n_layers (int): Number of LSTM layers.
            output_dim (int): Output dimension.
            dropout (float): Dropout probability.
            output_type (str): 'prediction' or 'embedding'.
            apply_softmax (bool): Whether to apply softmax to output.
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

    def forward(self, x, return_embedding=None, return_sequence=False):
        """
        Forward pass.

        Args:
            x (Tensor): Input sequence [batch, seq_len, input_dim].
            return_embedding (bool, optional): Override output type.
            return_sequence (bool, optional): If True, return full sequence.

        Returns:
            Tensor: Output [batch, output_dim] or [batch, hidden_dim] or sequence.
        """
        # h0, c0 initialized to zeros by default if not provided
        out, (h_n, c_n) = self.lstm(x)

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
            return state

        output = self.fc(state)

        if self.apply_softmax and self.output_type == "prediction":
            import torch.nn.functional as F

            output = F.softmax(output, dim=-1)

        return output


class GRU(nn.Module):
    """
    Gated Recurrent Unit (GRU) network for sequence processing.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_layers,
        output_dim,
        dropout=0.0,
        output_type="prediction",
    ):
        """
        Initialize the GRU.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden state dimension.
            n_layers (int): Number of GRU layers.
            output_dim (int): Output dimension.
            dropout (float): Dropout probability.
            output_type (str): 'prediction' or 'embedding'.
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

    def forward(self, x, return_embedding=None, return_sequence=False):
        """
        Forward pass.

        Args:
            x (Tensor): Input sequence [batch, seq_len, input_dim].
            return_embedding (bool, optional): Override output type.
            return_sequence (bool, optional): If True, return full sequence.

        Returns:
            Tensor: Output [batch, output_dim] or [batch, hidden_dim] or sequence.
        """
        # h0 initialized to zeros by default if not provided
        out, h_n = self.gru(x)

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
            return state

        return self.fc(state)
