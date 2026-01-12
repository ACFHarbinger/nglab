"""
Unified Backbone for Time Series Models.
"""
import torch.nn as nn
from .nstransformer import NSTransformer 

# Assuming NSTransformer exists in .nstransformer or we port it here.
# For now, we wrap it or define standard interfaces.

class TimeSeriesBackbone(nn.Module):
    """
    Unified Backbone for Time Series.
    Wraps specific implementations (Transformer, LSTM, etc).
    """
    def __init__(self, cfg):
        """
        Initialize the backbone.

        Args:
            cfg (dict): Configuration dictionary containing model name and parameters.
        """
        super().__init__()
        self.cfg = cfg
        model_name = cfg.get('name', 'NSTransformer')
        
        if model_name == 'NSTransformer':
            # Instantiate NSTransformer with cfg params
            # Placeholder instantiation logic matching NSTransformer signature
            self.model = NSTransformer(
                 pred_len=cfg.get('pred_len', 1),
                 seq_len=cfg.get('seq_len', 30),
                 input_dim=cfg.get('feature_dim', 12),
                 embed_dim=cfg.get('embed_dim', 64),
                 hidden_dim=cfg.get('hidden_dim', 128),
                 output_dim=cfg.get('output_dim', 64), # Output of encoder
                 learner_dims=cfg.get('learner_dims', [64]),
                 # ... other params ...
            )
        elif model_name == 'LSTM':
             self.model = nn.LSTM(
                 input_size=cfg.get('feature_dim', 12),
                 hidden_size=cfg.get('hidden_dim', 128),
                 num_layers=cfg.get('num_layers', 2),
                 batch_first=True
             )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor or TensorDict): Input data.

        Returns:
            Tensor: Output embeddings.
        """
        # Handle TensorDict input if x is TensorDict
        if hasattr(x, 'get'): 
            x = x.get('observation')
            
        # Forward pass through backbone
        # Ensure output is consistent (e.g. embeddings)
        out = self.model(x)
        
        # If LSTM, it returns (out, (h, c))
        if isinstance(out, tuple):
            out = out[0]
            # Take last step or avg? usually last step for RL/Classif
            out = out[:, -1, :] 
            
        return out
