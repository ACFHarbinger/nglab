"""
Extreme Learning Machine (ELM) implementation.
"""

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class ELM(nn.Module):
    """
    Extreme Learning Machine (ELM).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation='sigmoid', output_type='prediction'):
        """Initialize ELM."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_type = output_type
        
        self.register_buffer('w', torch.randn(hidden_dim, input_dim))
        self.register_buffer('b', torch.randn(hidden_dim))
        
        self.act_fn = {
            'sigmoid': torch.sigmoid,
            'relu': torch.relu,
            'tanh': torch.tanh
        }.get(activation.lower(), torch.sigmoid)
        
        self.readout = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, return_embedding=None, return_sequence=False):
        """Forward pass."""
        if x.dim() == 3:
            b, s, f = x.shape
            x_flat = x.view(b * s, f)
            h = self.act_fn(F.linear(x_flat, self.w, self.b))
            out = self.readout(h)
            h = h.view(b, s, -1)
            out = out.view(b, s, -1)
        else:
            h = self.act_fn(F.linear(x, self.w, self.b))
            out = self.readout(h)
            
        should_return_embedding = return_embedding if return_embedding is not None else (self.output_type == 'embedding')
        
        if should_return_embedding:
            res = h
        else:
            res = out
            
        if not return_sequence and res.dim() == 3:
            return res[:, -1, :]
        return res
