"""
Neural Turing Machine (NTM) - Neural network with addressable memory
"""

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class NTMMemory(nn.Module):
    """
    External memory module for Neural Turing Machine.

    Implements content-based and location-based addressing mechanisms.

    Args:
        memory_size: Number of memory slots (N)
        memory_dim: Dimension of each memory slot (M)
    """

    def __init__(self, memory_size: int, memory_dim: int):
        """Initialize NTM Memory."""
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim

    def content_addressing(
        self, memory: torch.Tensor, key: torch.Tensor, strength: torch.Tensor
    ) -> torch.Tensor:
        """
        Content-based addressing using cosine similarity.

        w_c = softmax(β * K(k, M))

        Args:
            memory: Memory matrix (batch, N, M)
            key: Key vector (batch, M)
            strength: Key strength β (batch, 1)

        Returns:
            Content weights (batch, N)
        """
        # Cosine similarity
        key = key.unsqueeze(1)  # (batch, 1, M)
        memory_norm = F.normalize(memory, p=2, dim=2)
        key_norm = F.normalize(key, p=2, dim=2)

        similarity = torch.sum(memory_norm * key_norm, dim=2)  # (batch, N)

        # Apply strength and softmax
        weights = F.softmax(strength * similarity, dim=1)
        return weights

    def location_addressing(
        self,
        content_weights: torch.Tensor,
        gate: torch.Tensor,
        shift: torch.Tensor,
        sharpen: torch.Tensor,
        prev_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Location-based addressing with interpolation, shift, and sharpening.

        1. Interpolation: w_g = g*w_c + (1-g)*w_{t-1}
        2. Shift: w_s = circular_conv(w_g, s)
        3. Sharpen: w_t = w_s^gamma / sum(w_s^gamma)

        Args:
            content_weights: Content-based weights (batch, N)
            gate: Interpolation gate g ∈ [0,1] (batch, 1)
            shift: Shift distribution (batch, shift_range)
            sharpen: Sharpening factor gamma >= 1 (batch, 1)
            prev_weights: Previous weights (batch, N)

        Returns:
            Final addressing weights (batch, N)
        """
        # 1. Interpolation
        gated = gate * content_weights + (1 - gate) * prev_weights

        # 2. Convolutional shift
        shift_range = shift.size(1)
        shifted = self._convolutional_shift(gated, shift, shift_range)

        # 3. Sharpening
        sharpened = shifted**sharpen
        weights = sharpened / (sharpened.sum(dim=1, keepdim=True) + 1e-8)

        return weights

    def _convolutional_shift(
        self, weights: torch.Tensor, shift: torch.Tensor, shift_range: int
    ) -> torch.Tensor:
        """
        Perform circular convolution for shifting.

        Args:
            weights: Input weights (batch, N)
            shift: Shift distribution (batch, shift_range)
            shift_range: Range of shift (typically 3 for [-1, 0, +1])

        Returns:
            Shifted weights (batch, N)
        """
        _batch_size, N = weights.shape

        # Create shifted versions of weights
        # Pad weights circularly
        weights_padded = torch.cat(
            [weights[:, -shift_range // 2 :], weights, weights[:, : shift_range // 2]],
            dim=1,
        )

        # Perform convolution
        shifted = torch.zeros_like(weights)
        for i in range(shift_range):
            offset = i - shift_range // 2
            start_idx = shift_range // 2 + offset
            end_idx = start_idx + N
            shifted += shift[:, i : i + 1] * weights_padded[:, start_idx:end_idx]

        return shifted


class NTMReadHead(nn.Module):
    """
    NTM Read Head with content and location addressing.

    Args:
        memory_size: Number of memory slots
        memory_dim: Dimension of each memory slot
        controller_dim: Dimension of controller output
        shift_range: Range of allowed shifts (typically 3 for [-1, 0, +1])
    """

    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        controller_dim: int,
        shift_range: int = 3,
    ):
        """Initialize NTM Read Head."""
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.shift_range = shift_range

        # Parameters for addressing
        self.key_net = nn.Linear(controller_dim, memory_dim)
        self.strength_net = nn.Linear(controller_dim, 1)
        self.gate_net = nn.Linear(controller_dim, 1)
        self.shift_net = nn.Linear(controller_dim, shift_range)
        self.sharpen_net = nn.Linear(controller_dim, 1)

        self.memory = NTMMemory(memory_size, memory_dim)

    def forward(
        self,
        controller_output: torch.Tensor,
        memory: torch.Tensor,
        prev_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory using NTM addressing.

        Args:
            controller_output: Output from controller (batch, controller_dim)
            memory: Memory matrix (batch, N, M)
            prev_weights: Previous read weights (batch, N)

        Returns:
            Tuple of (read_vector, new_weights)
        """
        # Generate addressing parameters
        key = self.key_net(controller_output)
        strength = F.softplus(self.strength_net(controller_output)) + 1
        gate = torch.sigmoid(self.gate_net(controller_output))
        shift = F.softmax(self.shift_net(controller_output), dim=1)
        sharpen = F.softplus(self.sharpen_net(controller_output)) + 1

        # Content addressing
        content_weights = self.memory.content_addressing(memory, key, strength)

        # Location addressing
        weights = self.memory.location_addressing(
            content_weights, gate, shift, sharpen, prev_weights
        )

        # Read from memory
        read_vector = torch.sum(weights.unsqueeze(2) * memory, dim=1)

        return read_vector, weights


class NTMWriteHead(nn.Module):
    """
    NTM Write Head with content and location addressing.

    Args:
        memory_size: Number of memory slots
        memory_dim: Dimension of each memory slot
        controller_dim: Dimension of controller output
        shift_range: Range of allowed shifts
    """

    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        controller_dim: int,
        shift_range: int = 3,
    ):
        """Initialize NTM Write Head."""
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.shift_range = shift_range

        # Addressing parameters
        self.key_net = nn.Linear(controller_dim, memory_dim)
        self.strength_net = nn.Linear(controller_dim, 1)
        self.gate_net = nn.Linear(controller_dim, 1)
        self.shift_net = nn.Linear(controller_dim, shift_range)
        self.sharpen_net = nn.Linear(controller_dim, 1)

        # Write parameters
        self.erase_net = nn.Linear(controller_dim, memory_dim)
        self.add_net = nn.Linear(controller_dim, memory_dim)

        self.memory = NTMMemory(memory_size, memory_dim)

    def forward(
        self,
        controller_output: torch.Tensor,
        memory: torch.Tensor,
        prev_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Write to memory using NTM addressing.

        Args:
            controller_output: Output from controller (batch, controller_dim)
            memory: Memory matrix (batch, N, M)
            prev_weights: Previous write weights (batch, N)

        Returns:
            Tuple of (updated_memory, new_weights)
        """
        # Generate addressing parameters
        key = self.key_net(controller_output)
        strength = F.softplus(self.strength_net(controller_output)) + 1
        gate = torch.sigmoid(self.gate_net(controller_output))
        shift = F.softmax(self.shift_net(controller_output), dim=1)
        sharpen = F.softplus(self.sharpen_net(controller_output)) + 1

        # Write parameters
        erase = torch.sigmoid(self.erase_net(controller_output))
        add = self.add_net(controller_output)

        # Content addressing
        content_weights = self.memory.content_addressing(memory, key, strength)

        # Location addressing
        weights = self.memory.location_addressing(
            content_weights, gate, shift, sharpen, prev_weights
        )

        # Write to memory
        # Erase: M_t = M_{t-1} * (1 - w_t * e_t^T)
        erase_matrix = torch.einsum("bn,bm->bnm", weights, erase)
        memory = memory * (1 - erase_matrix)

        # Add: M_t = M_t + w_t * a_t^T
        add_matrix = torch.einsum("bn,bm->bnm", weights, add)
        memory = memory + add_matrix

        return memory, weights


class NTM(nn.Module):
    """
    Neural Turing Machine.

    A neural network with an external memory matrix that can be
    read from and written to using differentiable attention mechanisms.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension of controller
        memory_size: Number of memory slots
        memory_dim: Dimension of each memory slot
        num_reads: Number of read heads
        num_writes: Number of write heads
        output_dim: Output dimension
        controller_type: Type of controller ('lstm' or 'linear')
        output_type: 'prediction' returns output, 'embedding' returns controller state
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        hidden_dim: int = 128,
        memory_size: int = 128,
        memory_dim: int = 20,
        num_reads: int = 1,
        num_writes: int = 1,
        output_dim: int = 10,
        controller_type: Literal["lstm", "linear"] = "lstm",
        output_type: Literal["prediction", "embedding"] = "prediction",
    ):
        """Initialize NTM."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_reads = num_reads
        self.num_writes = num_writes
        self.output_dim = output_dim
        self.controller_type = controller_type
        self.output_type = output_type

        # Controller (receives input + read vectors)
        controller_input_dim = input_dim + num_reads * memory_dim

        self.controller: nn.Module
        if controller_type == "lstm":
            self.controller = nn.LSTM(
                controller_input_dim, hidden_dim, batch_first=True
            )
        else:
            self.controller = nn.Linear(controller_input_dim, hidden_dim)

        # Read heads
        self.read_heads = nn.ModuleList(
            [NTMReadHead(memory_size, memory_dim, hidden_dim) for _ in range(num_reads)]
        )

        # Write heads
        self.write_heads = nn.ModuleList(
            [
                NTMWriteHead(memory_size, memory_dim, hidden_dim)
                for _ in range(num_writes)
            ]
        )

        # Output network
        self.output_net = nn.Linear(hidden_dim + num_reads * memory_dim, output_dim)

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        Forward pass through NTM.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            return_sequence: If True, return output for each timestep

        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = x.shape

        # Initialize memory and weights
        memory = torch.zeros(
            batch_size, self.memory_size, self.memory_dim, device=x.device
        )
        memory = memory + 1e-6  # Small initialization to avoid NaN

        read_weights = [
            torch.zeros(batch_size, self.memory_size, device=x.device)
            for _ in range(self.num_reads)
        ]
        write_weights = [
            torch.zeros(batch_size, self.memory_size, device=x.device)
            for _ in range(self.num_writes)
        ]

        # Initialize read vectors
        read_vectors = torch.zeros(
            batch_size, self.num_reads * self.memory_dim, device=x.device
        )

        # Initialize controller state
        if self.controller_type == "lstm":
            h_state = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)
            c_state = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)

        outputs_list = []

        for t in range(seq_len):
            # Controller input: current input + previous reads
            controller_input = torch.cat([x[:, t, :], read_vectors], dim=1)

            # Controller forward
            if self.controller_type == "lstm":
                controller_input = controller_input.unsqueeze(1)
                controller_output, (h_state, c_state) = self.controller(
                    controller_input, (h_state, c_state)
                )
                controller_output = controller_output.squeeze(1)
            else:
                controller_output = self.controller(controller_input)
                controller_output = torch.relu(controller_output)

            # Read from memory
            read_vecs = []
            for i, read_head in enumerate(self.read_heads):
                read_vec, read_weights[i] = read_head(
                    controller_output, memory, read_weights[i]
                )
                read_vecs.append(read_vec)

            read_vectors = torch.cat(read_vecs, dim=1)

            # Write to memory
            for i, write_head in enumerate(self.write_heads):
                memory, write_weights[i] = write_head(
                    controller_output, memory, write_weights[i]
                )

            # Generate output
            output_input = torch.cat([controller_output, read_vectors], dim=1)
            output = self.output_net(output_input)
            outputs_list.append(output)

        outputs = torch.stack(outputs_list, dim=1)

        if self.output_type == "embedding":
            if return_sequence:
                return outputs
            else:
                return outputs[:, -1, :]

        if return_sequence:
            return outputs
        else:
            return outputs[:, -1, :]
