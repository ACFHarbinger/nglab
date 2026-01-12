"""
Time Series Mamba implementation.
"""
import torch
import torch.nn as nn

from modules import MambaBlock


class TSMamba(nn.Module):
    """
    Time Series Forecasting Model using Mamba Blocks.
    """
    def __init__(self, input_dim, output_dim, d_model=64, n_layers=2, forecast_horizon=1):
        """
        Initialize the TSMamba model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features per time step.
            d_model (int): Model hidden dimension.
            n_layers (int): Number of Mamba blocks.
            forecast_horizon (int): Prediction offset/horizon.
        """
        super().__init__()
        
        # Encoder to project continuous time series values to d_model
        self.encoder = nn.Linear(input_dim, d_model)
        
        # Stack Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model) for _ in range(n_layers)
        ])
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Forecasting Head (Project to output dimension)
        # Note: For multi-step, we predict a vector or use direct strategy 
        self.head = nn.Linear(d_model, output_dim * forecast_horizon)

    def forward(self, x):
        """
        Forward pass for forecasting.
        """
        # x shape: (Batch, Seq_Len, Input_Dim)
        x = self.encoder(x)
        
        for layer in self.layers:
            x = x + layer(x) # Residual connection per block
            
        x = self.norm(x)
        
        # We typically take the last state for forecasting
        last_state = x[:, -1, :] 
        
        prediction = self.head(last_state)
        return prediction



# --- Example Usage ---
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    SEQ_LEN = 96      # Lookback window
    INPUT_DIM = 1     # Univariate time series
    OUTPUT_DIM = 1    # Forecast one variable
    HORIZON = 24      # Forecast 24 steps ahead (Direct strategy)

    # Instantiate model
    model = TSMamba(
        input_dim=INPUT_DIM, 
        output_dim=OUTPUT_DIM, 
        d_model=64, 
        n_layers=2, 
        forecast_horizon=HORIZON
    )

    # Dummy Data (e.g., random noise)
    # Shape: (Batch, Lookback, Features)
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)

    # Forward pass
    forecast = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Forecast shape: {forecast.shape}") # Should be (32, 24)
    print("Mamba model initialized successfully.")