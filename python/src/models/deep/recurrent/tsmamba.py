"""
Time Series Mamba implementation.
"""

import torch
from torch import nn

from python.src.models.deep.modules.mamba_block import MambaBlock


class TSMamba(nn.Module):
    """
    Time Series Forecasting Model using Mamba Blocks.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim,
        output_dim,
        d_model=64,
        n_layers=2,
        forecast_horizon=1,
        dropout=0.0,
        output_type="prediction",
    ):
        """
        Initialize the TSMamba model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features per time step.
            d_model (int): Model hidden dimension.
            n_layers (int): Number of Mamba blocks.
            forecast_horizon (int): Prediction offset/horizon.
            dropout (float): Dropout rate.
            output_type (str): 'prediction' or 'embedding'.
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
        # Note: For multi-step, we predict a vector or use direct strategy
        self.head = nn.Linear(d_model, output_dim * forecast_horizon)

    def forward(self, x, return_embedding=None, return_sequence=False):
        """
        Forward pass for forecasting.

        Args:
            x (Tensor): Input tensor of shape (Batch, Seq_Len, Input_Dim)
            return_embedding (bool): Override output type. If True, return embedding.
            return_sequence (bool): If True, return full sequence.
        """
        # x shape: (Batch, Seq_Len, Input_Dim)
        x = self.encoder(x)

        for layer in self.layers:
            x = x + layer(x)  # Residual connection per block

        x = self.norm(x)

        # Determine state to use (Full sequence or Last step)
        if return_sequence:
            state = x
        else:
            state = x[:, -1, :]

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            return state

        # Apply head
        return self.head(state)


# --- Example Usage ---
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    SEQ_LEN = 96  # Lookback window
    INPUT_DIM = 1  # Univariate time series
    OUTPUT_DIM = 1  # Forecast one variable
    HORIZON = 24  # Forecast 24 steps ahead (Direct strategy)

    # Instantiate model
    model = TSMamba(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        d_model=64,
        n_layers=2,
        forecast_horizon=HORIZON,
    )

    # Dummy Data (e.g., random noise)
    # Shape: (Batch, Lookback, Features)
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)

    # Forward pass
    forecast = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Forecast shape: {forecast.shape}")  # Should be (32, 24)
    print("Mamba model initialized successfully.")
