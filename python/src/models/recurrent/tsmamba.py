"""
Time Series Mamba implementation.
"""

from typing import cast

import torch
from torch import nn

from python.src.models.modules.mamba_block import MambaBlock


class TSMamba(nn.Module):
    """
    Time Series Forecasting Model using Mamba Blocks.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        n_layers: int = 2,
        forecast_horizon: int = 1,
        dropout: float = 0.0,
        output_type: str = "prediction",
    ) -> None:
        """
        Initialize the TSMamba model.
        """
        super().__init__()
        self.output_type = output_type

        # Encoder to project continuous time series values to d_model
        self.encoder = nn.Linear(input_dim, d_model)

        # Stack Mamba blocks
        self.layers = nn.ModuleList(
            [
                MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
                for _ in range(n_layers)
            ]
        )

        # Normalization
        self.norm = nn.LayerNorm(d_model)

        # Forecasting Head (Project to output dimension)
        self.head = nn.Linear(d_model, output_dim * forecast_horizon)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for forecasting.

        Args:
            x (Tensor): Input tensor of shape (Batch, Seq_Len, Input_Dim)
            return_embedding (bool): Override output type. If True, return embedding.
            return_sequence (bool): If True, return full sequence.
        """
        # x shape: (Batch, Seq_Len, Input_Dim)
        x_enc = cast(torch.Tensor, self.encoder(x))

        for layer_module in self.layers:
            layer = cast(MambaBlock, layer_module)
            x_enc = x_enc + cast(
                torch.Tensor, layer(x_enc)
            )  # Residual connection per block

        x_norm = cast(torch.Tensor, self.norm(x_enc))

        # Determine state to use (Full sequence or Last step)
        state: torch.Tensor
        if return_sequence:
            state = x_norm
        else:
            state = x_norm[:, -1, :]

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            return state

        # Apply head
        return cast(torch.Tensor, self.head(state))
