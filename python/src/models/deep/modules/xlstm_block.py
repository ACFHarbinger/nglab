"""
xLSTM Block implementation (sLSTM variant).
"""

import torch
from torch import nn


class sLSTMCell(nn.Module):
    """
    Scalar LSTM (sLSTM) Cell with exponential gating and normalization.
    """

    def __init__(self, input_dim, hidden_dim):
        """Initialize sLSTM Cell."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Weights for [z, i, f, o]
        # z: input update (tanh)
        # i: input gate (exponential)
        # f: forget gate (exponential)
        # o: output gate (sigmoid)
        self.weight_ih = nn.Linear(input_dim, 4 * hidden_dim)
        self.weight_hh = nn.Linear(hidden_dim, 4 * hidden_dim)

    def forward(self, x, state):
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
        # Initial state handling needs to be done by caller or defaults here
        if state is None:
            # Requires batch size from x
            pass
        c_prev, n_prev, m_prev, h_prev = state

        # Project inputs
        gates = self.weight_ih(x) + self.weight_hh(h_prev)
        z_gate, i_gate, f_gate, o_gate = gates.chunk(4, dim=1)

        # Activations
        # z: standard input update
        z_t = torch.tanh(z_gate)

        # o: output gate (sigmoid)
        o_t = torch.sigmoid(o_gate)

        # Exponential Gating with Stabilization
        # i_gate and f_gate are in log-space effectively before exp
        # m_t = max(f_gate + m_{t-1}, i_gate)
        # Note: f_gate here acts as log(forget_factor). Standard GLU/LSTM uses sigmoid(f).
        # xLSTM uses exp(f) usually.

        # Stabilization logic:
        # We want to compute:
        # c_t = exp(f_gate)*c_{t-1} + exp(i_gate)*z_t
        # n_t = exp(f_gate)*n_{t-1} + exp(i_gate)
        # h_t = o_t * (c_t / n_t)

        # To avoid overflow, we track m_t (max log value seen so far)
        # red_f = f_gate + m_{t-1}
        # red_i = i_gate
        # m_t = max(red_f, red_i)

        # i_prime = exp(i_gate - m_t)
        # f_prime = exp(f_gate + m_{t-1} - m_t)

        m_t = torch.max(f_gate + m_prev, i_gate)

        i_prime = torch.exp(i_gate - m_t)
        f_prime = torch.exp(f_gate + m_prev - m_t)

        # State updates
        c_t = f_prime * c_prev + i_prime * z_t
        n_t = f_prime * n_prev + i_prime

        # Output updates
        # Check for division by zero? n_t shouldn't be zero if initialized properly (e.g. n=0, m large negative? or n=1?)
        # Convention: If n_t is small, stabilization helps. n_t is sum of exps, so always > 0.

        h_t = o_t * (c_t / (n_t + 1e-6))  # Small epsilon for safety

        return h_t, (c_t, n_t, m_t, h_t)


class mLSTMCell(nn.Module):
    """
    Matrix LSTM (mLSTM) Cell.
    Uses a matrix memory C_t (d x d) updated via outer product of keys and values.
    Equivalent to linear attention with a recurrence.
    """

    def __init__(self, input_dim, hidden_dim, num_heads=4):
        """Initialize mLSTM Cell."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        is_divisible = self.head_dim * num_heads == hidden_dim
        assert is_divisible, "Hidden dim must be divisible by num_heads"

        # Projects inputs to q, k, v and gates
        # We need:
        # q: (B, H, D_head)
        # k: (B, H, D_head)
        # v: (B, H, D_head)
        # i: (B, H) - input gate (scalar per head) or (B, H, 1)
        # f: (B, H) - forget gate (scalar per head)
        # o: (B, H, D_head) - output gate? Or just (B, Hidden)

        # Simplification: Projections for all
        # Input to Q, K, V, i_gate, f_gate, o_gate
        # Total dims:
        # q, k, v: 3 * hidden_dim
        # i, f: 2 * num_heads (scalars per head) or 2 * hidden_dim? Use head-wise gating usually for memory.
        # o: hidden_dim (element-wise output gate like LSTM)

        self.weight_ih = nn.Linear(
            input_dim, 3 * hidden_dim + 2 * num_heads + hidden_dim
        )
        # Ordering: q, k, v, i (heads), f (heads), o (hidden)

    def forward(self, x, state):
        """
        Forward step.
        State: (C, n, m)
           C: (B, H, D_head, D_head) - Matrix state
           n: (B, H, D_head) - Normalizer state
           m: (B, H, 1) - Max state for stabilization
        """
        batch_size = x.size(0)

        # Projections
        # No recurrence weight_hh in standard Transformer/Linear Attention sense usually?
        # But xLSTM paper keeps it recurrent?
        # "sLSTM uses recurrence, mLSTM enhances with matrix memory".
        # xLSTM blocks usually just project input x (which might come from a mixing layer).
        # We'll assume just input projection here for the core cell. The Block adds residual/mixing.

        projected = self.weight_ih(x)  # (B, 3*H*D + 2*H + H*D)

        # Split
        base_split = [self.hidden_dim] * 3  # q, k, v
        gate_split = [self.num_heads] * 2  # i, f scalars
        out_split = [self.hidden_dim]  # o

        splits = projected.split(base_split + gate_split + out_split, dim=-1)
        q, k, v, i_gate, f_gate, o_gate = splits

        # Reshape for heads
        # q, k, v: (B, Heads, Head_Dim)
        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)

        # Gates: (B, Heads, 1)
        i_gate = i_gate.view(batch_size, self.num_heads, 1)
        f_gate = f_gate.view(batch_size, self.num_heads, 1)

        o_gate = torch.sigmoid(o_gate)  # (B, Hidden) or (B, H, D)
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

        # Stabilization logic (similar to sLSTM but separate m_t for memory)
        # m_t = max(f + m_{t-1}, i)
        m_t = torch.max(f_gate + m_prev, i_gate)

        i_prime = torch.exp(i_gate - m_t)  # (B, H, 1)
        f_prime = torch.exp(f_gate + m_prev - m_t)  # (B, H, 1)

        # Update C (Matrix memory)
        # C_t = f' * C_{t-1} + i' * (v @ k.T)
        # k: (B, H, D) -> (B, H, D, 1)
        # v: (B, H, D) -> (B, H, D, 1)
        # kvT: (B, H, D, 1) @ (B, H, 1, D) -> (B, H, D, D)
        kvT = torch.matmul(v.unsqueeze(-1), k.unsqueeze(-2))

        C_t = f_prime.unsqueeze(-1) * C_prev + i_prime.unsqueeze(-1) * kvT

        # Update n (Normalizer)
        # n_t = f' * n_{t-1} + i' * k
        n_t = f_prime * n_prev + i_prime * k

        # Retrieval / Output calculation
        # h_tilde = C_t @ q / |n_t @ q| ?
        # Standard: h = (C_t @ q) / max(|n_t . q|, 1)
        # q: (B, H, D, 1)

        q_uns = q.unsqueeze(-1)
        num = torch.matmul(C_t, q_uns).squeeze(-1)  # (B, H, D)

        # den = n_t . k ??? No, n_t is accumulating k's.
        # Wait, linear attention normalization is usually: Sum(k). This is n_t.
        # Then we divide by n_t . q? No, that's not right for standard linear attention.
        # Standard Linear Attn: O = (V K^T) Q / (1 K^T) Q ?
        # xLSTM paper: n_t = f' n_{t-1} + i' k
        # denominator = |n_t^T q| (dot product)

        den = torch.sum(n_t * q, dim=-1, keepdim=True)  # Dot product per head
        den = torch.abs(den)  # Manual says "absolute value" usually or max(den, eps)

        h_tilde = num / (den + 1e-6)

        # Gating
        h_t = o_gate_view * h_tilde
        h_t = h_t.reshape(batch_size, self.hidden_dim)  # Flatten heads

        return h_t, (C_t, n_t, m_t)


class xLSTMBlock(nn.Module):
    """
    xLSTM Block layer (wrapping sLSTMCell or mLSTMCell).
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        batch_first=True,
        dropout=0.0,
        cell_type="slstm",
        num_heads=4,
    ):
        """Initialize xLSTM Block."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.cell_type = cell_type.lower()

        if self.cell_type == "slstm":
            self.cell = sLSTMCell(input_dim, hidden_dim)
        elif self.cell_type == "mlstm":
            self.cell = mLSTMCell(input_dim, hidden_dim, num_heads=num_heads)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, state=None):
        """
        Process sequence.
        x: (Batch, Seq, Feat) if batch_first
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # -> (Seq, Batch, Feat)

        seq_len, _batch_size, _ = x.shape

        outputs = []
        current_state = state

        # Loop implementation
        # (Could be optimized with scan or custom kernel)
        for t in range(seq_len):
            inp = x[t]
            h_t, current_state = self.cell(inp, current_state)
            h_t = self.dropout(h_t)
            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=0)  # (Seq, Batch, Hidden)

        if self.batch_first:
            outputs = outputs.transpose(0, 1)

        return outputs, current_state
