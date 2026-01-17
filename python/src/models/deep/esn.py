"""
Echo State Network (ESN) implementation.
"""

import torch
import torch
import torch.nn as nn

class EchoStateNetwork(nn.Module):
    """
    Echo State Network (ESN).
    """
    def __init__(self, input_dim, reservoir_dim, output_dim, spectral_radius=0.9, sparsity=0.1, output_type='prediction'):
        """
        Initialize ESN.
        """
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.output_type = output_type
        
        self.register_buffer('win', (torch.rand(reservoir_dim, input_dim) - 0.5) * 2.0)
        w_res = (torch.rand(reservoir_dim, reservoir_dim) - 0.5) * 2.0
        mask = torch.rand(reservoir_dim, reservoir_dim) < sparsity
        w_res = w_res * mask.float()
        
        eigenvalues = torch.linalg.eigvals(w_res)
        max_eigen = torch.max(torch.abs(eigenvalues))
        w_res = w_res * (spectral_radius / max_eigen)
        self.register_buffer('wres', w_res)
        
        self.readout = nn.Linear(reservoir_dim + input_dim, output_dim)
        
    def forward(self, x, return_embedding=None, return_sequence=False):
        """Calculate forward pass."""
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.reservoir_dim, device=x.device)
        
        states = []
        outputs = []
        for t in range(seq_len):
            u = x[:, t, :]
            h = torch.tanh(torch.matmul(u, self.win.t()) + torch.matmul(h, self.wres.t()))
            
            combined = torch.cat([u, h], dim=1)
            out = self.readout(combined)
            states.append(h)
            outputs.append(out)
            
        should_return_embedding = return_embedding if return_embedding is not None else (self.output_type == 'embedding')
        
        if should_return_embedding:
            res = torch.stack(states, dim=1)
        else:
            res = torch.stack(outputs, dim=1)
            
        if not return_sequence:
            return res[:, -1, :]
        return res
