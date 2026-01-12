"""Static and Dynamic Hyper-Network connections."""
import torch
import torch.nn as nn


class StaticHyperConnection(nn.Module):
    """
    Hyper-connection with static width/depth expansion.
    """
    def __init__(self, module: nn.Module, hyper_dim: int, expansion_rate: int = 4):
        """
        Initializes the static hyper-connection.

        Args:
            module: The neural network module to wrap.
            hyper_dim: Hyper-network dimension.
            expansion_rate: Width/depth expansion rate.
        """
        super().__init__()
        self.module = module
        self.n = expansion_rate
        
        # Initialize slightly off-identity to preserve gradient flow at start
        self.width_mixer = nn.Parameter(torch.eye(self.n) + torch.randn(self.n, self.n) * 0.01)
        self.input_mixer = nn.Parameter(torch.randn(self.n, 1) * 0.01)
        self.depth_mixer = nn.Parameter(torch.randn(1, self.n) * 0.01)

    def forward(self, H, *args, **kwargs):
        """
        Forward pass for static hyper connection.
        """
        # H shape: (Batch, Seq, Dim, n)
        
        # 1. Collapse streams for the sub-layer (A_m)
        # (B, S, D, n) x (n, 1) -> (B, S, D, 1) -> (B, S, D)
        h_in = torch.matmul(H, self.input_mixer).squeeze(-1)
        
        # 2. Apply Sub-layer (Attention, MLP, etc.)
        y = self.module(h_in, *args, **kwargs) # (B, S, D)
        
        # 3. Update Hyper Matrix
        # Width: Mix existing streams (H x A_r)
        term_width = torch.matmul(H, self.width_mixer)
        
        # Depth: Broadcast new info (y x B)
        term_depth = torch.matmul(y.unsqueeze(-1), self.depth_mixer)
        
        return term_width + term_depth


class DynamicHyperConnection(nn.Module):
    """
    Dynamic Hyper-Connection Network (DyHCN).
    
    Dynamically generates connection weights based on input context using a predictor network.
    Uses 'streams' (n) concepts to mix information.
    """
    def __init__(self, module, embed_dim, n=4):
        """
        Initializes the dynamic hyper-connection.

        Args:
            module: The sub-module to wrap.
            embed_dim: Embedding dimension.
            n: Number of streams.
        """
        super(DynamicHyperConnection, self).__init__()
        self.module = module
        self.n = n
        self.embed_dim = embed_dim
        
        # Calculate total parameters needed for matrices: n*n (width) + n (input) + n (depth)
        self.num_params = (self.n * self.n) + self.n + self.n
        
        # Predictor Network: Maps input embedding -> Matrix Weights
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, self.num_params)
        )

        self._initialize_identity_bias()
    
    def _initialize_identity_bias(self):
        """
        Forces the predictor to output Identity for Width and Zero for others initially.
        """
        # Set weights to near-zero so bias dominates at initialization
        nn.init.uniform_(self.predictor[-1].weight, -0.0001, 0.0001)
        
        n = self.n
        # Width Mixer (A_r) target: Flattened Identity Matrix
        target_width = torch.eye(n).flatten()
        # Input (A_m) and Depth (B) targets: Zeros (or small random)
        target_others = torch.zeros(2 * n)
        
        initial_bias = torch.cat([target_width, target_others])
        self.predictor[-1].bias.data.copy_(initial_bias)

    def forward(self, H, *args, **kwargs):
        """
        Applies the dynamic hyper-connection.

        Args:
            H: Input tensor of shape (Batch, Seq, Dim, n).
            *args: Additional arguments for the sub-module.
            **kwargs: Additional keyword arguments for the sub-module.

        Returns:
            Updated tensor.
        """
        # H shape: (Batch, Seq, Dim, n)
        
        # 1. Generate a proxy input for the predictor (e.g., mean of streams)
        x_proxy = H.mean(dim=-1) # (B, S, D)
        
        # 2. Predict Weights for this specific input
        # params: (B, S, num_params)
        params = self.predictor(x_proxy) 
        
        # Split params into A_r, A_m, B
        n = self.n
        B, S, _ = params.shape
        
        # Width (n*n), Input (n), Depth (n)
        P_width = params[..., :n*n].view(B, S, n, n)     # (B, S, n, n)
        P_input = params[..., n*n:n*n+n].view(B, S, n, 1) # (B, S, n, 1)
        P_depth = params[..., n*n+n:].view(B, S, 1, n)    # (B, S, 1, n)
        
        # 3. Collapse streams (Dynamic A_m)
        # H: (B, S, D, n) -> (B, S, n, D) for matmul with (B, S, n, 1) if doing elementwise
        # Easier approach: Use manual einsum or simple broadcasting
        # H (B,S,D,n) * P_input (B,S,1,n,1) -> tricky broadcasting. 
        # Let's use einsum for clarity.
        
        # Input Mixer: sum_k (H_{ijk} * P_input_{ij k})
        # b:batch, s:seq, d:dim, n:streams
        h_in = torch.einsum('bsdn,bsnk->bsd', H, P_input)
        
        # 4. Apply Sub-layer
        y = self.module(h_in, *args, **kwargs)
        
        # 5. Update Hyper Matrix
        # Width: H x P_width
        term_width = torch.einsum('bsdn,bsnm->bsdm', H, P_width)
        
        # Depth: y x P_depth
        term_depth = torch.einsum('bsd,bsmn->bsdn', y, P_depth)
        
        return term_width + term_depth
