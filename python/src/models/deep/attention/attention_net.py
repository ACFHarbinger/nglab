"""
Attention Network - Self-attention mechanism for sequence modeling
"""

import math
from typing import Literal, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        dropout: Dropout probability for attention weights
    """

    def __init__(self, dropout: float = 0.1):
        """Initialize Scaled Dot-Product Attention."""
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor (batch, ..., seq_len, d_k)
            key: Key tensor (batch, ..., seq_len, d_k)
            value: Value tensor (batch, ..., seq_len, d_v)
            mask: Optional mask tensor

        Returns:
            Tuple of (attention_output, attention_weights)
        """
        d_k = query.size(-1)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, value)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """Initialize Multi-Head Attention."""
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Output projection
        self.out_linear = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional mask tensor

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size = query.size(0)

        # Linear projections and split into heads
        Q = (
            self.q_linear(query)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.k_linear(key)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.v_linear(value)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Apply attention
        attn_output, _ = self.attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        # Final linear projection
        output = self.out_linear(attn_output)
        output = self.dropout(output)

        return cast(torch.Tensor, output)


class AttentionBlock(nn.Module):
    """
    Attention Block with feed-forward network and residual connections.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
    """

    def __init__(
        self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1
    ):
        """Initialize Attention Block."""
        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.ff_network(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class AttentionNetwork(nn.Module):
    """
    Attention Network for sequence modeling.

    A stack of attention blocks that process sequences using
    self-attention mechanisms.

    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        num_layers: Number of attention blocks
        num_heads: Number of attention heads per block
        d_ff: Feed-forward dimension
        output_dim: Output dimension
        dropout: Dropout probability
        max_seq_len: Maximum sequence length (for positional encoding)
        output_type: 'prediction' returns final output, 'embedding' returns features
    """

    positional_encoding: torch.Tensor

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 512,
        output_dim: int = 10,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        output_type: Literal["prediction", "embedding"] = "prediction",
    ):
        """
        Initialize Attention Network.
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.output_type = output_type

        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.register_buffer(
            "positional_encoding",
            self._generate_positional_encoding(max_seq_len, d_model),
        )

        # Stack of attention blocks
        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)

        self.dropout = nn.Dropout(dropout)

    def _generate_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Generate sinusoidal positional encodings.

        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Args:
            max_len: Maximum sequence length
            d_model: Model dimension

        Returns:
            Positional encoding tensor (max_len, d_model)
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through Attention Network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            return_sequence: If True, return output for each timestep

        Returns:
            Output tensor of shape (batch_size, output_dim) or (batch_size, seq_len, output_dim)
        """
        _batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = self.dropout(x)

        # Pass through attention blocks
        for block in self.attention_blocks:
            x = block(x, mask)

        # Layer normalization
        x = self.norm(x)

        if self.output_type == "embedding":
            if return_sequence:
                return x
            else:
                # Return last timestep or pooled representation
                return x[:, -1, :]

        # Output projection
        out = self.output_projection(x)

        if return_sequence:
            return cast(torch.Tensor, out)
        else:
            # Return last timestep
            return cast(torch.Tensor, out[:, -1, :])

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create a causal mask for autoregressive modeling.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Causal mask tensor (seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
