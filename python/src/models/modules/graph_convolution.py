"""Standard Graph Convolutional Network (GCN) layer."""
import math
import torch
import torch.nn as nn

from torch_geometric.utils import scatter


class GraphConvolution(nn.Module):
    """
    Standard Graph Convolution layer.
    
    Performs message passing by aggregating features from neighbors:
    h_i' = W * h_i + Agg({h_j})
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 aggregation: str = "sum",
                 bias: bool = True):
        """
        Args:
            in_channels: Dimension of input node features.
            out_channels: Dimension of output node features.
            aggregation: Aggregation method ('sum', 'mean', 'max').
            bias: Whether to include a learnable bias term.
        """
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregation = aggregation
        
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_parameters()
        
    def init_parameters(self):
        """Initializes the parameters of the layer using uniform distribution."""
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
            
    def forward(self, h, mask):
        """ 
        Args:
            h: Input node features (B x V x H_1)
            mask: Graph adjacency matrix (V x V) - shared across all batches
            
        Returns:
            Updated node features (B x V x H_2)
        """
        batch_size, num_nodes = h.size(0), h.size(1)
        support = self.linear(h)
        if self.aggregation == "max":
            source_idx, target_idx = mask.squeeze(0).nonzero(as_tuple=True)
            num_edges = source_idx.size(0)
            batch_indices = torch.arange(batch_size, device=h.device)
            
            expanded_source = source_idx.repeat(batch_size)
            expanded_target = target_idx.repeat(batch_size)
            batch_id = batch_indices.repeat_interleave(num_edges)
            
            # Get messages from source nodes for all batches
            messages = support[batch_id, expanded_source]
            batch_offset = batch_id * num_nodes
            unique_targets = batch_offset + expanded_target
            
            # Use scatter to aggregate messages to targets
            flat_output = scatter(messages, unique_targets, dim=0, 
                                dim_size=batch_size * num_nodes, reduce="max")
            out = flat_output.view(batch_size, num_nodes, self.out_channels)
        elif self.aggregation == "mean":
            # Check if mask is batched (B, V, V) or shared (1, V, V) / (V, V)
            if mask.dim() == 3:
                degrees = mask.sum(dim=-1, keepdim=True).clamp(min=1)
                # degrees is (B, V, 1)
                if degrees.size(0) == 1: # Shared graph but 3D
                    degrees = degrees.expand(batch_size, -1, -1)
            else: # (V, V)
                degrees = mask.sum(dim=-1).view(1, num_nodes, 1).expand(batch_size, -1, -1).clamp(min=1)
            
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            elif mask.size(0) == 1:
                mask = mask.expand(batch_size, -1, -1)
                
            out = torch.bmm(mask, support)
            out = out / degrees
        else:
            out = torch.bmm(mask.expand(batch_size, -1, -1), support)
        
        if self.bias is not None:
            out += self.bias
        return out
    
    def single_graph_forward(self, h, adj):
        """        
        Args:
            h: Input node features (N x H_1)
            adj: Graph adjacency matrix (N x N)
            
        Returns:
            Updated node features (N x H_2)
        """
        support = self.linear(h)
        if self.aggregation == "max":
            source_idx, target_idx = adj.nonzero(as_tuple=True)
            messages = support[source_idx]
            out = scatter(messages, target_idx, dim=0, dim_size=h.size(0), reduce="max")
        elif self.aggregation == "mean":
            degrees = adj.sum(dim=0).clamp(min=1)
            messages = torch.matmul(adj, support)
            out = messages / degrees.unsqueeze(1)
        else: 
            out = torch.matmul(adj, support)
        
        if self.bias is not None:
            out += self.bias
        return out