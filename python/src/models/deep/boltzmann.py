"""
Boltzmann Machine (BM) implementation.
"""

import torch
import torch
import torch.nn as nn

class BoltzmannMachine(nn.Module):
    """
    Boltzmann Machine (BM) - Stochastic recurrent neural network with symmetric connections.
    Implements Gibbs sampling and energy-based learning.
    Full connectivity between all units.
    """
    def __init__(self, num_units, output_type='prediction'):
        """Initialize Boltzmann Machine."""
        super().__init__()
        self.num_units = num_units
        self.output_type = output_type
        
        # Symmetric weight matrix with zero diagonal
        self.weights = nn.Parameter(torch.randn(num_units, num_units) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_units))
        
    def get_weights(self):
        """Get symmetric weights with zero diagonal."""
        # Force symmetry and zero diagonal
        w = (self.weights + self.weights.t()) / 2
        return w.fill_diagonal_(0)
        
    def forward(self, x, iterations=10, return_embedding=None, return_sequence=False):
        """
        Gibbs sampling for state evolution.
        x: (Batch, Num_Units) or (Batch, Seq, Num_Units)
        """
        should_return_embedding = return_embedding if return_embedding is not None else (self.output_type == 'embedding')
        
        if x.dim() == 3:
            b, s, f = x.shape
            x_flat = x.view(b * s, f)
            res = self._gibbs_sampling(x_flat, iterations)
            res = res.view(b, s, -1)
        else:
            res = self._gibbs_sampling(x, iterations)
            
        if not return_sequence and res.dim() == 3:
            return res[:, -1, :]
        return res

    def _gibbs_sampling(self, state, iterations):
        w = self.get_weights()
        for _ in range(iterations):
            # Sigmoid(Wx + b)
            probs = torch.sigmoid(torch.matmul(state, w) + self.bias)
            state = torch.bernoulli(probs)
        return state
        
    def energy(self, state):
        """Energy function: -0.5 * x^T W x - b^T x"""
        w = self.get_weights()
        # (Batch, 1, Num_Units) @ (Num_Units, Num_Units) @ (Batch, Num_Units, 1)
        e = -0.5 * torch.bmm(state.unsqueeze(1), torch.matmul(w, state.unsqueeze(2))).squeeze()
        e -= torch.matmul(state, self.bias)
        return e
