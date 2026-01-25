"""
Time Series GAN Networks (Generator and Discriminator).
"""

from typing import cast

import torch
from torch import nn


class TimeGANGenerator(nn.Module):
    """
    Generator for Time Series Forecasting GAN.
    Seq2Seq LSTM Architecture: Encoder -> Decoder.
    Takes history X_{1:t} and generates X_{t+1:t+k}.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        pred_len: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ) -> None:
        """
        Initialize the Generator.
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(output_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (Batch, Seq_Len, Features)
        Returns: (Batch, Pred_Len, Features)
        """
        # Encode
        _, (h_n, c_n) = self.encoder(x)

        # Let's seed decoder with last observation
        curr_input = x[:, -1:, :]  # (B, 1, F)

        outputs: list[torch.Tensor] = []

        # State
        h_state, c_state = h_n, c_n

        for _ in range(self.pred_len):
            out, (h_state, c_state) = self.decoder(curr_input, (h_state, c_state))
            # out: (B, 1, Hidden)
            pred = self.fc(out)  # (B, 1, Out_F)
            outputs.append(pred)
            curr_input = pred  # Autoregressive interaction

        result = torch.cat(outputs, dim=1)  # (B, Pred_Len, F)
        return result


class TimeGANDiscriminator(nn.Module):
    """
    Discriminator for Time Series GAN.
    Takes full sequence (History + Future) and classifies as Real/Fake.
    Architecture: Bidirectional LSTM.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 2) -> None:
        """
        Initialize the Discriminator.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (Batch, Seq_Len + Pred_Len, Features)
        Returns: (Batch, 1) - Logits
        """
        output, _ = self.lstm(x)  # (B, L, 2*H)

        # Global Average Pooling over time to catch anomalies anywhere
        out_pooled = torch.mean(output, dim=1)
        logits = self.fc(out_pooled)
        return cast(torch.Tensor, logits)
