"""
xLSTM Block implementation (sLSTM variant).
"""

import torch
from torch import nn


class sLSTMCell(nn.Module):  # noqa: N801
    """
    Scalar LSTM (sLSTM) Cell with exponential gating and normalization.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initialize sLSTM Cell."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Weights for [z, i, f, o]
        self.weight_ih = nn.Linear(input_dim, 4 * hidden_dim)
        self.weight_hh = nn.Linear(hidden_dim, 4 * hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Forward pass for a single time step.

        Args:
            x: Input tensor (Batch, Input_Dim)
            state: Tuple of (c, n, m, h)
                   c: Cell state (Batch, Hidden_Dim)
                   n: Normalizer state (Batch, Hidden_Dim)
                   m: Stabilizer state (Batch, Hidden_Dim) - log scale
                   h: Hidden state (Batch, Hidden_Dim)
        """
        # Unpack state
        c_prev, n_prev, m_prev, h_prev = state

        # Project inputs
        gates = self.weight_ih(x) + self.weight_hh(h_prev)
        z_gate, i_gate, f_gate, o_gate = gates.chunk(4, dim=1)

        # Activations
        z_t = torch.tanh(z_gate)
        o_t = torch.sigmoid(o_gate)

        m_t = torch.max(f_gate + m_prev, i_gate)

        i_prime = torch.exp(i_gate - m_t)
        f_prime = torch.exp(f_gate + m_prev - m_t)

        # State updates
        c_t = f_prime * c_prev + i_prime * z_t
        n_t = f_prime * n_prev + i_prime

        h_t = o_t * (c_t / (n_t + 1e-6))  # Small epsilon for safety

        return h_t, (c_t, n_t, m_t, h_t)


class mLSTMCell(nn.Module):  # noqa: N801
    """
    Matrix LSTM (mLSTM) Cell.
    Uses a matrix memory C_t (d x d) updated via outer product of keys and values.
    Equivalent to linear attention with a recurrence.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4) -> None:
        """Initialize mLSTM Cell."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        is_divisible = self.head_dim * num_heads == hidden_dim
        assert is_divisible, "Hidden dim must be divisible by num_heads"

        self.weight_ih = nn.Linear(
            input_dim, 3 * hidden_dim + 2 * num_heads + hidden_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward step.
        State: (C, n, m)
           C: (B, H, D_head, D_head) - Matrix state
           n: (B, H, D_head) - Normalizer state
           m: (B, H, 1) - Max state for stabilization
        """
        batch_size = x.size(0)

        projected = self.weight_ih(x)  # (B, 3*H*D + 2*H + H*D)

        # Split
        base_split = [self.hidden_dim] * 3  # q, k, v
        gate_split = [self.num_heads] * 2  # i, f scalars
        out_split = [self.hidden_dim]  # o

        splits = projected.split(base_split + gate_split + out_split, dim=-1)
        q, k, v, i_gate, f_gate, o_gate = splits

        # Reshape for heads
        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)

        # Gates: (B, Heads, 1)
        i_gate = i_gate.view(batch_size, self.num_heads, 1)
        f_gate = f_gate.view(batch_size, self.num_heads, 1)

        o_gate = torch.sigmoid(o_gate)  # (B, Hidden)
        o_gate_view = o_gate.view(batch_size, self.num_heads, self.head_dim)

        # State unpacking
        if state is None:
            # Init state
            dev = x.device
            C_prev = torch.zeros(
                batch_size, self.num_heads, self.head_dim, self.head_dim, device=dev
            )
            n_prev = torch.zeros(batch_size, self.num_heads, self.head_dim, device=dev)
            m_prev = torch.zeros(batch_size, self.num_heads, 1, device=dev)
        else:
            C_prev, n_prev, m_prev = state

        m_t = torch.max(f_gate + m_prev, i_gate)

        i_prime = torch.exp(i_gate - m_t)  # (B, H, 1)
        f_prime = torch.exp(f_gate + m_prev - m_t)  # (B, H, 1)

        kvT = torch.matmul(v.unsqueeze(-1), k.unsqueeze(-2))
        C_t = f_prime.unsqueeze(-1) * C_prev + i_prime.unsqueeze(-1) * kvT
        n_t = f_prime * n_prev + i_prime * k

        q_uns = q.unsqueeze(-1)
        num = torch.matmul(C_t, q_uns).squeeze(-1)  # (B, H, D)

        den = torch.sum(n_t * q, dim=-1, keepdim=True)  # Dot product per head
        den = torch.abs(den)

        h_tilde = num / (den + 1e-6)

        # Gating
        h_t = o_gate_view * h_tilde
        h_t = h_t.reshape(batch_size, self.hidden_dim)  # Flatten heads

        return h_t, (C_t, n_t, m_t)


class xLSTMBlock(nn.Module):  # noqa: N801
    """
    xLSTM Block layer (wrapping sLSTMCell or mLSTMCell).
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        hidden_dim: int,
        batch_first: bool = True,
        dropout: float = 0.0,
        cell_type: str = "slstm",
        num_heads: int = 4,
    ) -> None:
        """Initialize xLSTM Block."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.cell_type = cell_type.lower()

        if self.cell_type == "slstm":
            self.cell: nn.Module = sLSTMCell(input_dim, hidden_dim)
        elif self.cell_type == "mlstm":
            self.cell = mLSTMCell(input_dim, hidden_dim, num_heads=num_heads)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, ...] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """
        Process sequence.
        x: (Batch, Seq, Feat) if batch_first
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # -> (Seq, Batch, Feat)

        seq_len, batch_size, _ = x.shape

        outputs = []
        current_state = state

        # Initial state for sLSTM if None
        if current_state is None and self.cell_type == "slstm":
            dev = x.device
            current_state = (
                torch.zeros(batch_size, self.hidden_dim, device=dev),
                torch.zeros(batch_size, self.hidden_dim, device=dev),
                torch.full(
                    (batch_size, self.hidden_dim), -100.0, device=dev
                ),  # log scale
                torch.zeros(batch_size, self.hidden_dim, device=dev),
            )

        for t in range(seq_len):
            inp = x[t]
            h_t, current_state = self.cell(inp, current_state)
            h_t = self.dropout(h_t)
            outputs.append(h_t)

        result_outputs = torch.stack(outputs, dim=0)  # (Seq, Batch, Hidden)

        if self.batch_first:
            result_outputs = result_outputs.transpose(0, 1)

        return result_outputs, current_state
