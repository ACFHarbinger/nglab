"""Gated Graph Convolution (GatedGCN) with explicit edge updates."""
import math
import torch
import torch.nn as nn

from . import Normalization, ActivationFunction


class GatedGraphConvolution(nn.Module):
    """Implements the Gated Graph ConvNet layer:
        h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
        sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
        e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
        where Aggr. is an aggregation function: sum/mean/max.

    References:
        - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. In International Conference on Learning Representations, 2018.
        - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
    """
    """
    Gated Graph Convolution Layer (GatedGCN).
    
    Explicitly models edge features and uses a gating mechanism to control
    information flow between neighbors. Updates both node and edge features.
    """
    def __init__(self, 
                hidden_dim: int,
                aggregation: str = "sum",
                norm: str = "batch",
                activation: str = "relu",
                learn_affine: bool = True,
                gated: bool = True,
                bias: bool = True):
        """
        Args:
            hidden_dim: Dimension of hidden features (input and output are assumed same size).
            aggregation: Aggregation method ('sum', 'mean', 'max').
            norm: Normalization type ('batch', 'layer', 'instance').
            activation: Activation function name.
            learn_affine: Whether normalization layers have learnable affine parameters.
            gated: Whether to use the gating mechanism.
            bias: Whether to use a bias term in linear layers.
        """
        super(GatedGraphConvolution, self).__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm = norm
        self.learn_affine = learn_affine
        self.gated = gated
        assert self.gated, "Use gating with GCN, pass the `--gated` flag"
        
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.norm_h = Normalization(hidden_dim, self.norm, learn_affine)
        self.norm_e = Normalization(hidden_dim, self.norm, learn_affine)
        self.activation = ActivationFunction(activation)
    
    def reset_parameters(self):
        """Initializes the parameters of the layer using uniform distribution."""
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
        
    def forward(self, h, e, mask):
        """
        Args:
            h: Input node features (B x V x H)
            e: Input edge features (B x V x V x H)
            mask: Graph adjacency matrices (B x V x V)
        Returns: 
            Updated node and edge features
        """
        batch_size, num_nodes, hidden_dim = h.shape
        #h_in = h
        #e_in = e

        # Linear transformations for node update
        Uh = self.U(h)  # B x V x H
        Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

        # Linear transformations for edge update and gating
        Ah = self.A(h)  # B x V x H
        Bh = self.B(h)  # B x V x H
        Ce = self.C(e)  # B x V x V x H

        # Update edge features and compute edge gates
        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
        gates = torch.sigmoid(e)  # B x V x V x H

        # Update node features
        h = Uh + self.aggregate(Vh, mask, gates)  # B x V x H

        # Normalize node features
        h = self.norm_h(h) if self.norm_h else h
        
        # Normalize edge features
        # For edges, we need to flatten B, V, V. Normalization handles 3D/4D if mapped correctly.
        # Normalization(hidden_dim) expects (N, H) or (B, N, H).
        # For edge (B, V, V, H), we view as (B, V*V, H)
        B, V, _, H = e.shape
        e = self.norm_e(e.view(B, -1, H)).view(B, V, V, H) if self.norm_e else e

        # Apply non-linearity
        h = self.activation(h)
        e = self.activation(e)

        return h, e

    def aggregate(self, Vh, mask, gates):
        """
        Args:
            Vh: Neighborhood features (B x V x V x H)
            gates: Edge gates (B x V x V x H)
            mask: Graph adjacency matrices (B x V x V)
        Returns:
            Aggregated neighborhood features (B x V x H)
        """
        # Perform feature-wise gating mechanism
        Vh = gates * Vh  # B x V x V x H
        
        # Enforce graph structure through masking
        Vh[mask.unsqueeze(-1).expand_as(Vh)] = 0
        
        if self.aggregation == "mean":
            return torch.sum(Vh, dim=2) / torch.sum(1-mask, dim=2).unsqueeze(-1).type_as(Vh)
        elif self.aggregation == "max":
            return torch.max(Vh, dim=2)[0]
        else:
            return torch.sum(Vh, dim=2)