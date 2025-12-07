import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """
    A single Mamba block that implements the Selection Mechanism and SSM.
    Ref: Section 4.3.1 of the provided text.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_model / 16)

        # 1. Input Projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # 2. Convolution (1D) - captures local context before the SSM
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # 3. Selection Mechanism Projections
        # Projects input x to delta, B, and C parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # Projects delta to the correct dimension
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 4. S4D (Structured State Space) Parameters
        # A_log is learnable, initializing the state transition matrix A
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 5. Output Projection
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def parallel_scan_dummy(self, u, delta, A, B, C):
        """
        A simplified sequential implementation of the SSM equation:
        h_t = A_t * h_{t-1} + B_t * x_t
        y_t = C_t * h_t
        
        Note: Production Mamba uses a hardware-aware parallel scan here.
        """
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[-1]
        
        # Discretize A (continuous -> discrete) using Zero-Order Hold (ZOH)
        # delta shape: (batch, seq, d_inner)
        # A shape: (d_inner, d_state) -> broadcast to (batch, seq, d_inner, d_state)
        delta_A = torch.exp(torch.einsum('b l d, d n -> b l d n', delta, A))
        delta_B = torch.einsum('b l d, b l n -> b l d n', delta, B)
        
        # Initial state
        h = torch.zeros(batch_size, d_inner, d_state, device=u.device)
        ys = []
        
        # Sequential scan (The loop that replaces O(L^2) attention)
        for t in range(seq_len):
            # h_t = A_bar * h_{t-1} + B_bar * u_t
            h = delta_A[:, t] * h + delta_B[:, t] * u[:, t].unsqueeze(-1)
            
            # y_t = C_t * h_t
            y = torch.einsum('b d n, b n -> b d', h, C[:, t])
            ys.append(y)
            
        return torch.stack(ys, dim=1) 

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 1. Project inputs
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        (x_branch, res_branch) = x_and_res.split(self.d_inner, dim=-1)

        # 2. Convolution (Transpose for Conv1d: B, D, L)
        x_branch = x_branch.permute(0, 2, 1)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len] # Crop padding
        x_branch = x_branch.permute(0, 2, 1)
        x_branch = F.silu(x_branch)

        # 3. SSM / Selection Mechanism
        # Derive discrete parameters from the input (data-dependent) 
        x_dbl = self.x_proj(x_branch) # (B, L, dt_rank + 2*d_state)
        
        (delta, B, C) = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        delta = F.softplus(self.dt_proj(delta)) # Softplus ensures positive time-step
        A = -torch.exp(self.A_log) # Keep A negative for stability

        # Run the SSM
        y = self.parallel_scan_dummy(x_branch, delta, A, B, C)
        
        # Residual connection + Gating
        y = y + x_branch * self.D
        y = y * F.silu(res_branch) # Gating mechanism
        
        return self.out_proj(y)