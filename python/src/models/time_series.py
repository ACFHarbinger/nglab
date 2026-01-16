"""
Unified Backbone for Time Series Models.
"""
import torch.nn as nn
from .nstransformer import NSTransformer 
from .tsmamba import TSMamba
from .rnn import LSTM, GRU
from .xlstm import xLSTM
from .snn import SNN
from .mlp import MLP
from .rbf import RBF
from .ae import AutoEncoder
from .dae import DenoisingAE
from .sae import SparseAE
from .hopfield import HopfieldNetwork
from .rbm import RBM
from .esn import EchoStateNetwork
from .elm import ELM
from .som import KohonenMap
from .capsule import CapsuleLayer
from .cnn import RollingWindowCNN

class TimeSeriesBackbone(nn.Module):
    """
    Unified Backbone for Time Series.
    Wraps specific implementations (Transformer, LSTM, etc).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_name = cfg.get('name', 'NSTransformer')
        
        if model_name == 'NSTransformer':
            self.model = NSTransformer(
                 pred_len=cfg.get('pred_len', 1),
                 seq_len=cfg.get('seq_len', 30),
                 input_dim=cfg.get('feature_dim', 12),
                 embed_dim=cfg.get('embed_dim', 64),
                 hidden_dim=cfg.get('hidden_dim', 128),
                 output_dim=cfg.get('output_dim', 64),
                 learner_dims=cfg.get('learner_dims', [64]),
            )
        elif model_name == 'Mamba':
             self.model = TSMamba(
                 input_dim=cfg.get('feature_dim', 12),
                 output_dim=1,
                 d_model=cfg.get('hidden_dim', 128),
                 n_layers=cfg.get('num_layers', 2),
                 forecast_horizon=cfg.get('pred_len', 1),
                 output_type=cfg.get('output_type', 'embedding')
             )
        elif model_name == 'LSTM':
             self.model = LSTM(
                 input_dim=cfg.get('feature_dim', 12),
                 output_dim=cfg.get('output_dim', 1),
                 hidden_dim=cfg.get('hidden_dim', 128),
                 n_layers=cfg.get('num_layers', 2),
                 dropout=cfg.get('dropout', 0.0),
                 output_type=cfg.get('output_type', 'embedding')
             )
        elif model_name == 'GRU':
             self.model = GRU(
                 input_dim=cfg.get('feature_dim', 12),
                 output_dim=cfg.get('output_dim', 1),
                 hidden_dim=cfg.get('hidden_dim', 128),
                 n_layers=cfg.get('num_layers', 2),
                 dropout=cfg.get('dropout', 0.0),
                 output_type=cfg.get('output_type', 'embedding')
             )
        elif model_name == 'xLSTM':
             self.model = xLSTM(
                 input_dim=cfg.get('feature_dim', 12),
                 output_dim=1,
                 hidden_dim=cfg.get('hidden_dim', 128),
                 n_layers=cfg.get('num_layers', 2),
                 dropout=cfg.get('dropout', 0.0),
                 output_type=cfg.get('output_type', 'embedding'),
                 cell_type=cfg.get('cell_type', 'slstm'),
                 num_heads=cfg.get('num_heads', 4)
             )
        elif model_name == 'SNN':
             self.model = SNN(
                 input_dim=cfg.get('feature_dim', 12),
                 output_dim=cfg.get('output_dim', 1),
                 hidden_dim=cfg.get('hidden_dim', 128),
                 n_layers=cfg.get('num_layers', 2),
                 dropout=cfg.get('dropout', 0.0),
                 output_type=cfg.get('output_type', 'embedding'),
                 decay=cfg.get('decay', 0.9),
                 threshold=cfg.get('threshold', 1.0)
             )
        elif model_name == 'MLP':
             self.model = MLP(
                 input_dim=cfg.get('feature_dim', 12),
                 hidden_dims=cfg.get('hidden_dims', [128, 64]),
                 output_dim=cfg.get('output_dim', 1),
                 dropout=cfg.get('dropout', 0.0),
                 activation=cfg.get('activation', 'relu'),
                 output_type=cfg.get('output_type', 'embedding')
             )
        elif model_name == 'RBF':
             self.model = RBF(
                 input_dim=cfg.get('feature_dim', 12),
                 num_centers=cfg.get('hidden_dim', 100),
                 output_dim=cfg.get('output_dim', 1),
                 sigma=cfg.get('sigma', 1.0),
                 output_type=cfg.get('output_type', 'embedding')
             )
        elif model_name == 'AE':
             self.model = AutoEncoder(
                 input_dim=cfg.get('feature_dim', 12),
                 hidden_dims=cfg.get('hidden_dims', [64]),
                 latent_dim=cfg.get('hidden_dim', 32),
                 output_type=cfg.get('output_type', 'embedding')
             )
        elif model_name == 'DAE':
             self.model = DenoisingAE(
                 input_dim=cfg.get('feature_dim', 12),
                 hidden_dims=cfg.get('hidden_dims', [64]),
                 latent_dim=cfg.get('hidden_dim', 32),
                 noise_std=cfg.get('noise_std', 0.1),
                 output_type=cfg.get('output_type', 'embedding')
             )
        elif model_name == 'SAE':
             self.model = SparseAE(
                 input_dim=cfg.get('feature_dim', 12),
                 hidden_dims=cfg.get('hidden_dims', [64]),
                 latent_dim=cfg.get('hidden_dim', 32),
                 sparsity_target=cfg.get('sparsity_target', 0.05),
                 sparsity_weight=cfg.get('sparsity_weight', 0.1),
                 output_type=cfg.get('output_type', 'embedding')
             )
        elif model_name == 'Hopfield':
             self.model = HopfieldNetwork(
                 size=cfg.get('feature_dim', 12),
                 output_type=cfg.get('output_type', 'embedding')
             )
        elif model_name == 'RBM':
             self.model = RBM(
                 visible_dim=cfg.get('feature_dim', 12),
                 hidden_dim=cfg.get('hidden_dim', 64),
                 output_type=cfg.get('output_type', 'embedding')
             )
        elif model_name == 'ESN':
             self.model = EchoStateNetwork(
                 input_dim=cfg.get('feature_dim', 12),
                 reservoir_dim=cfg.get('hidden_dim', 500),
                 output_dim=cfg.get('output_dim', 1),
                 spectral_radius=cfg.get('spectral_radius', 0.9),
                 sparsity=cfg.get('sparsity', 0.1),
                 output_type=cfg.get('output_type', 'embedding')
             )
        elif model_name == 'ELM':
             self.model = ELM(
                 input_dim=cfg.get('feature_dim', 12),
                 hidden_dim=cfg.get('hidden_dim', 500),
                 output_dim=cfg.get('output_dim', 1),
                 activation=cfg.get('activation', 'sigmoid'),
                 output_type=cfg.get('output_type', 'embedding')
             )
        elif model_name == 'SOM':
             self.model = KohonenMap(
                 input_dim=cfg.get('feature_dim', 12),
                 grid_size=cfg.get('grid_size', (10, 10)),
                 output_type=cfg.get('output_type', 'embedding')
             )
        elif model_name == 'Capsule':
             self.model = CapsuleLayer(
                 in_caps=cfg.get('in_caps', 8),
                 in_dim=cfg.get('in_dim', 16),
                 out_caps=cfg.get('out_caps', 4),
                 out_dim=cfg.get('out_dim', 32),
                 output_type=cfg.get('output_type', 'embedding')
             )
        elif model_name == 'CNN':
             self.model = RollingWindowCNN(
                 input_dim=cfg.get('feature_dim', 12),
                 output_dim=1,
                 seq_len=cfg.get('seq_len', 30),
                 hidden_dim=cfg.get('hidden_dim', 128),
                 output_type=cfg.get('output_type', 'embedding')
             )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def forward(self, x):
        if hasattr(x, 'get'): 
            x = x.get('observation')
            
        kwargs = {}
        if self.cfg.get('return_sequence', False):
             kwargs['return_sequence'] = True
        
        # Whitelist for models supporting return_sequence
        # All our new models I just refactored support it.
        sequence_supported = [
            'LSTM', 'GRU', 'Mamba', 'xLSTM', 'SNN', 'ESN', 'MLP', 'ELM', 
            'RBF', 'AE', 'DAE', 'SAE', 'Hopfield', 'RBM', 'SOM', 'Capsule'
        ]
        
        if self.cfg.get('name') in sequence_supported:
             pass
        else:
             if 'return_sequence' in kwargs:
                 del kwargs['return_sequence']
        
        out = self.model(x, **kwargs)
        return out
