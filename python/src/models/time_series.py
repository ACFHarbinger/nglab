"""
Unified Backbone for Time Series Models.
"""
import torch.nn as nn
from .nstransformer import NSTransformer 
from .tsmamba import TSMamba
from .rnn import LSTM, GRU
from .xlstm import xLSTM
from .cnn import RollingWindowCNN

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
        elif model_name == 'Mamba':
             self.model = TSMamba(
                 input_dim=cfg.get('feature_dim', 12),
                 output_dim=1, # Unused in embedding mode
                 d_model=cfg.get('hidden_dim', 128),
                 n_layers=cfg.get('num_layers', 2),
                 forecast_horizon=cfg.get('pred_len', 1),
                 output_type='embedding'
             )
        elif model_name == 'LSTM':
             self.model = LSTM(
                 input_dim=cfg.get('feature_dim', 12),
                 output_dim=1, # Unused in embedding mode
                 hidden_dim=cfg.get('hidden_dim', 128),
                 n_layers=cfg.get('num_layers', 2),
                 dropout=cfg.get('dropout', 0.0),
                 output_type='embedding'
             )
        elif model_name == 'GRU':
             self.model = GRU(
                 input_dim=cfg.get('feature_dim', 12),
                 output_dim=1, # Unused in embedding mode
                 hidden_dim=cfg.get('hidden_dim', 128),
                 n_layers=cfg.get('num_layers', 2),
                 dropout=cfg.get('dropout', 0.0),
                 output_type='embedding'
             )
        elif model_name == 'xLSTM':
             self.model = xLSTM(
                 input_dim=cfg.get('feature_dim', 12),
                 output_dim=1, # Unused in embedding mode
                 hidden_dim=cfg.get('hidden_dim', 128),
                 n_layers=cfg.get('num_layers', 2),
                 dropout=cfg.get('dropout', 0.0),
                 output_type='embedding',
                 cell_type=cfg.get('cell_type', 'slstm'),
                 num_heads=cfg.get('num_heads', 4)
             )
        elif model_name == 'CNN':
             self.model = RollingWindowCNN(
                 input_dim=cfg.get('feature_dim', 12),
                 output_dim=1, # Unused in embedding mode
                 seq_len=cfg.get('seq_len', 30),
                 hidden_dim=cfg.get('hidden_dim', 128),
                 output_type='embedding'
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
        # All our wrapped models (NSTransformer, TSMamba, LSTM, GRU) now handle returning embedding/features
        # depending on initialization or direct call. NSTransformer returns features by default in encoder mode?
        # Check NSTransformer call: output_dim usually implies features.
        
        # Wrapped models (LSTM/GRU/Mamba) act as backbone with output_type='embedding'
        out = self.model(x)
        
        # Legacy check if we were using raw nn.LSTM which returned tuple
        # But now we use wrappers that return tensor.
        if isinstance(out, tuple):
            out = out[0]
            # Take last step or avg? usually last step for RL/Classif
            out = out[:, -1, :] 
            
        return out
