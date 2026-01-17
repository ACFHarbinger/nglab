"""
Hopfield Network implementation.
"""

import torch
import torch
import torch.nn as nn

class HopfieldNetwork(nn.Module):
    """
    Discrete Hopfield Network.
    """
    def __init__(self, size, output_type='embedding'):
        """Initialize Hopfield Network."""
        super().__init__()
        self.size = size
        self.output_type = output_type
        self.register_buffer('weights', torch.zeros(size, size))
        
    def store_patterns(self, patterns):
        """Store patterns using Hebbian learning."""
        w = torch.matmul(patterns.t(), patterns) / self.size
        w.fill_diagonal_(0)
        self.weights = w
        
    def forward(self, x, iterations=10, return_embedding=None, return_sequence=False):
        """
        Retrieval as a 'forward' pass.
        """
        # Handle sequence
        if x.dim() == 3:
            b, s, f = x.shape
            x_flat = x.view(b * s, f)
            y_flat = self._retrieve(x_flat, iterations)
            out = y_flat.view(b, s, -1)
        else:
            out = self._retrieve(x, iterations)
            
        if not return_sequence and out.dim() == 3:
            return out[:, -1, :]
        return out

    def _retrieve(self, x, iterations):
        s = x
        for _ in range(iterations):
            s = torch.sign(torch.matmul(s, self.weights))
            s[s == 0] = 1
        return s
