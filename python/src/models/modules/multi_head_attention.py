"""Multi-Head Attention mechanism for transformer architectures."""
import math
import torch
import torch.nn as nn
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Allows the model to jointly attend to information from different representation subspaces
    at different positions. Implements scaled dot-product attention.
    """
    def __init__(self, 
                n_heads:int,
                input_dim:int,
                embed_dim: Optional[int]=None,
                val_dim: Optional[int]=None,
                key_dim: Optional[int]=None):
        """
        Args:
            n_heads: Number of attention heads.
            input_dim: Dimensionality of the input features.
            embed_dim: Total embedding dimension (output dimension).
            val_dim: Dimension of value vectors per head.
            key_dim: Dimension of key/query vectors per head.
        """
        super(MultiHeadAttention, self).__init__()
        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
        self.init_parameters()
        self.last_attn = (None, None)

    def init_parameters(self):
        """Initializes the parameters of the attention layers using Xavier uniform initialization."""
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        Computes the multi-head attention.

        Args:
            q: Queries tensor of shape (batch_size, n_query, input_dim).
            h: Key/Value keys tensor of shape (batch_size, graph_size, input_dim). 
               If None, defaults to q (self-attention).
            mask: Attention mask of shape (batch_size, n_query, graph_size).
                  Should contain 1 (or > -inf) if attention is not possible, usually used as additive mask.

        Returns:
            context: Context vectors (aggregated values) of shape (batch_size, n_query, embed_dim).
            attn: Attention weights (batch_size, n_heads, n_query, graph_size).
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)  # [batch_size * graph_size, embed_dim]
        qflat = q.contiguous().view(-1, input_dim)  # [batch_size * n_query, embed_dim]

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            if mask.dim() == 2:
                # (batch, graph) -> (1, batch, n_query, graph) broadcastable to (heads, batch, query, graph)
                mask = mask.view(1, batch_size, n_query, graph_size)
            else:
                # Original logic assumption: mask is capable of expanding to (batch, -1, -1) and then view(1, batch, n_q, graph)
                # If user provides (batch, n_query, graph), expand(batch, -1, -1) works if n_query matches?
                # If mask is already compatible, we leave it or reshape carefully.
                # Given original code was: mask.expand(batch_size, -1, -1).view(1, batch_size, n_query, graph_size).expand_as(compatibility)
                # We try to respect it but make it safer.
                # If mask is 3D (batch, n_query, graph) or (1, 1, graph)?
                pass

            # Safe expand
            mask = mask.expand_as(compatibility)
            compatibility[mask] = -math.inf

        attn = torch.softmax(compatibility, dim=-1)  # [n_heads, batch_size, n_query, graph_size+1+n_pick*2] (graph_size include depot)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            self.last_attn = (attn.detach().clone(), mask.detach().clone())
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc
        else:
            self.last_attn = (attn.detach().clone(), mask)
        
        # heads: [n_heads, batrch_size, n_query, val_size], attn????pick?deliver?attn
        heads = torch.matmul(attn[:, :, :, :graph_size], V)  # V: (self.n_heads, batch_size, graph_size, val_size)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)
        return out