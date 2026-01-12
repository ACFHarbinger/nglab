"""Graph Convolutional layer with distance-aware edge weighting."""
import math
import torch
import torch.nn as nn

from torch_geometric.utils import scatter


class DistanceAwareGraphConvolution(nn.Module):
    """
    Graph Convolution that incorporates distance information into the message passing.
    
    Supports different aggregation methods (sum, mean, max) and handles
    edge features related to physical distance.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        distance_influence: str = "inverse",  # Options: "inverse", "exponential", "learnable"
        aggregation: str = "sum",
        bias: bool = True
    ):
        """
        Args:
            in_channels: Dimension of input node features.
            out_channels: Dimension of output node features.
            distance_influence: Method to incorporate distance ('inverse', 'exponential', 'learnable').
            aggregation: Aggregation method ('sum', 'mean', 'max').
            bias: Whether to use a bias term.
        """
        super(DistanceAwareGraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregation = aggregation
        self.distance_influence = distance_influence
        
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        
        # Additional parameters for distance weighting
        if distance_influence == "learnable":
            # Learn a function to transform distances into weights
            self.distance_transform = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        elif distance_influence == "exponential":
            # Parameter for exponential decay
            self.temp = nn.Parameter(torch.tensor(1.0))
        elif distance_influence == "inverse":
            # Parameter for inverse distance scaling
            self.alpha = nn.Parameter(torch.tensor(1.0))
            
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.init_parameters()
        
    def init_parameters(self):
        """Initializes the parameters of the layer using uniform distribution."""
        for param in self.parameters():
            if param.dim() > 1:
                stdv = 1. / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)
            
    def get_distance_weights(self, dist_matrix):
        """
        Compute edge weights based on distances
        
        Args:
            dist_matrix: Distance matrix (V x V)
            
        Returns:
            Edge weights with same shape as dist_matrix
        """
        # Avoid division by zero (self-loops have 0 distance)
        eps = 1e-8
        
        if self.distance_influence == "inverse":
            # Inverse distance weighting: 1/(d^alpha + eps)
            weights = 1.0 / (dist_matrix.pow(self.alpha) + eps)
            
        elif self.distance_influence == "exponential":
            # Exponential decay: exp(-d/temp)
            weights = torch.exp(-dist_matrix / self.temp.clamp(min=eps))
            
        elif self.distance_influence == "learnable":
            # Apply learned function to transform distances to weights
            # Reshape for the network
            flat_dist = dist_matrix.view(-1, 1)
            # Apply transformation
            flat_weights = self.distance_transform(flat_dist)
            # Reshape back
            weights = flat_weights.view(dist_matrix.shape)
            
        else:
            # Default: binary adjacency (no distance weighting)
            weights = (dist_matrix > 0).float()
            
        return weights
    
    def forward(self, h, adj, dist_matrix=None):
        """
        Args:
            h: Input node features (B x V x H_1)
            adj: Graph adjacency matrix (V x V) - shared across all batches
            dist_matrix: Distance matrix (V x V) - shared across all batches
            
        Returns:
            Updated node features (B x V x H_2)
        """
        batch_size, num_nodes = h.size(0), h.size(1)
        
        # Transform node features
        support = self.linear(h)
        
        # If no distance matrix is provided, use adjacency matrix
        if dist_matrix is None:
            dist_matrix = adj
            
        # Get distance-based edge weights
        edge_weights = self.get_distance_weights(dist_matrix)
        
        # Apply edge weights to adjacency matrix
        weighted_adj = adj * edge_weights
        
        if self.aggregation == "max":
            # For max aggregation with distance weighting
            source_idx, target_idx = adj.squeeze(0).nonzero(as_tuple=True)
            edge_weights_flat = edge_weights.view(-1)[adj.view(-1).nonzero().squeeze()]
            
            num_edges = source_idx.size(0)
            batch_indices = torch.arange(batch_size, device=h.device)
            expanded_source = source_idx.repeat(batch_size)
            expanded_target = target_idx.repeat(batch_size)
            batch_id = batch_indices.repeat_interleave(num_edges)
            
            # Get messages from source nodes for all batches
            messages = support[batch_id, expanded_source]
            
            # Weight messages by distance
            expanded_weights = edge_weights_flat.repeat(batch_size)
            weighted_messages = messages * expanded_weights.unsqueeze(1)
            
            batch_offset = batch_id * num_nodes
            unique_targets = batch_offset + expanded_target
            
            # Use scatter to aggregate messages to targets
            flat_output = scatter(weighted_messages, unique_targets, dim=0,
                                dim_size=batch_size * num_nodes, reduce="max")
            out = flat_output.view(batch_size, num_nodes, self.out_channels)
            
        if self.aggregation == "mean":
            # Calculate weighted node degrees for normalization
            # Handle batched/unbatched weighted_adj
            if weighted_adj.dim() == 3: # (B, N, N)
                 degrees = weighted_adj.sum(dim=-1).clamp(min=1e-8) # (B, N)
                 degrees = degrees.unsqueeze(-1) # (B, N, 1)
            else: # (N, N)
                 degrees = weighted_adj.sum(dim=-1).clamp(min=1e-8) # (N)
                 degrees = degrees.view(1, num_nodes, 1).expand(batch_size, -1, -1)

            # Weighted message passing
            if weighted_adj.dim() == 2:
                weighted_adj = weighted_adj.expand(batch_size, -1, -1)
            
            out = torch.bmm(weighted_adj, support)
            
            # Normalize by weighted degrees
            out = out / degrees
            
        else:  # Default: sum aggregation
            # Weighted message passing
            out = torch.bmm(weighted_adj.expand(batch_size, -1, -1), support)
        
        if self.bias is not None:
            out += self.bias
            
        return out
    
    def single_graph_forward(self, h, adj, dist_matrix=None):
        """
        Args:
            h: Input node features (N x H_1)
            adj: Graph adjacency matrix (N x N)
            dist_matrix: Distance matrix (N x N)
            
        Returns:
            Updated node features (N x H_2)
        """
        # Transform node features
        support = self.linear(h)
        
        # If no distance matrix is provided, use adjacency matrix
        if dist_matrix is None:
            dist_matrix = adj
            
        # Get distance-based edge weights
        edge_weights = self.get_distance_weights(dist_matrix)
        
        # Apply edge weights to adjacency matrix
        weighted_adj = adj * edge_weights
        
        if self.aggregation == "max":
            # For max aggregation with distance weighting
            source_idx, target_idx = adj.nonzero(as_tuple=True)
            edge_weights_flat = edge_weights.view(-1)[adj.view(-1).nonzero().squeeze()]
            
            # Get messages from source nodes
            messages = support[source_idx]
            
            # Weight messages by distance
            weighted_messages = messages * edge_weights_flat.unsqueeze(1)
            
            # Use scatter to aggregate messages to targets
            out = scatter(weighted_messages, target_idx, dim=0, 
                          dim_size=h.size(0), reduce="max")
            
        elif self.aggregation == "mean":
            # Calculate weighted node degrees for normalization
            degrees = weighted_adj.sum(dim=0).clamp(min=1e-8)
            
            # Weighted message passing
            messages = torch.matmul(weighted_adj, support)
            
            # Normalize by weighted degrees
            out = messages / degrees.unsqueeze(1)
            
        else:  # Default: sum aggregation
            # Weighted message passing
            out = torch.matmul(weighted_adj, support)
        
        if self.bias is not None:
            out += self.bias
            
        return out

# Example of how to use this class:
if __name__ == "__main__":
    # Example with batch of 2 graphs, each with 4 nodes and 3 features
    batch_size, num_nodes, in_features, out_features = 2, 4, 3, 5
    node_features = torch.randn(batch_size, num_nodes, in_features)
    
    # Create adjacency matrix (shared across batch)
    adj = torch.zeros(num_nodes, num_nodes)
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1  # For undirected graph
    
    # Create distance matrix
    dist_matrix = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Example distance - in a real scenario, this would be your actual distances
                dist_matrix[i, j] = abs(i - j)
    
    # Initialize and run the model
    for mode in ["inverse", "exponential", "learnable"]:
        model = DistanceAwareGraphConvolution(
            in_channels=in_features,
            out_channels=out_features,
            distance_influence=mode,
            aggregation="sum"
        )
        
        # Forward pass with distance information
        output = model(node_features, adj.unsqueeze(0), dist_matrix)
        print(f"Mode: {mode}, Output shape: {output.shape}")