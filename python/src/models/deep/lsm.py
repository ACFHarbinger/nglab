"""
Liquid State Machine (LSM) model.
"""

import torch
from torch import nn

from .snn import LIFCell, surrogate_heaviside


class LiquidStateMachine(nn.Module):
    """
    Reservoir computing with spiking neurons.
    Fixed sparse recurrent connections (liquid).
    Trainable readout layers.
    """

    def __init__(
        self,
        input_dim,
        liquid_size,
        output_dim,
        connection_prob=0.3,
        spectral_radius=1.2,
        output_type="prediction",
    ):
        """
        Initialize the Liquid State Machine.

        Args:
            input_dim: Input feature dimension.
            liquid_size: Size of the reservoir (liquid).
            output_dim: Output dimension.
            connection_prob: Probability of recurrent connections.
            spectral_radius: Spectral radius for weight scaling.
            output_type: Output type ('prediction' or 'embedding').
        """
        super().__init__()
        self.input_dim = input_dim
        self.liquid_size = liquid_size
        self.output_type = output_type

        # Input to Liquid (fixed random)
        self.register_buffer("win", torch.randn(liquid_size, input_dim) * 0.1)

        # Liquid Recurrent (fixed random sparse)
        w_liquid = torch.randn(liquid_size, liquid_size) * 0.1
        mask = torch.rand(liquid_size, liquid_size) < connection_prob
        w_liquid = w_liquid * mask.float()

        # Rescale for spectral radius
        eig = torch.linalg.eigvals(w_liquid).abs()
        if eig.max() > 0:
            w_liquid = w_liquid * (spectral_radius / eig.max())

        self.register_buffer("wliquid", w_liquid)

        # Spiking cell reference (mostly for parameters like decay/threshold)
        self.liquid_cell = LIFCell(input_dim, liquid_size)

        # Freeze parameters
        for p in self.parameters():
            p.requires_grad = False

        # Trainable readout
        self.readout = nn.Linear(liquid_size, output_dim)

    def forward(self, x, return_embedding=None, return_sequence=False):
        """
        x: (Batch, Seq, Feat)
        """
        batch_size, seq_len, _ = x.shape
        state = None  # (u, s)
        outputs = []

        device = x.device

        for t in range(seq_len):
            u_in = x[:, t, :]

            if state is None:
                mem = torch.zeros(batch_size, self.liquid_size, device=device)
                spk = torch.zeros(batch_size, self.liquid_size, device=device)
            else:
                mem, spk = state
                # Ensure they are seen as tensors
                assert isinstance(mem, torch.Tensor)
                assert isinstance(spk, torch.Tensor)

            # Recurrent injection: W_liquid @ previous spikes
            i_rec = torch.mm(spk, self.wliquid.t())

            # Input injection: W_in @ input
            i_inj = torch.mm(u_in, self.win.t())

            # Potential update: decay * (mem - spk * threshold) + i_inj + i_rec
            mem = (
                self.liquid_cell.decay * (mem - spk * self.liquid_cell.threshold)
                + i_inj
                + i_rec
            )

            # Fire
            spk = surrogate_heaviside(
                mem - self.liquid_cell.threshold, self.liquid_cell.alpha
            )
            state = (mem, spk)
            outputs.append(spk)

        stacked = torch.stack(outputs, dim=1)  # (Batch, Seq, Liquid_Size)

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            if return_sequence:
                return stacked
            return stacked[:, -1, :]

        out = self.readout(stacked)
        if return_sequence:
            return out
        return out[:, -1, :]
