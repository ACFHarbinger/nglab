"""
Differentiable Neural Computer (DNC) - Neural network with external memory
"""

from typing import Literal, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class DNCMemory(nn.Module):
    """
    External memory module for DNC.

    Implements content-based addressing, temporal memory linkage,
    and allocation/deallocation mechanisms.

    Args:
        memory_size: Number of memory slots (N)
        memory_dim: Dimension of each memory slot (W)
    """

    def __init__(self, memory_size: int, memory_dim: int) -> None:
        """Initialize DNC Memory."""
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim

    def content_addressing(
        self, memory: torch.Tensor, key: torch.Tensor, strength: torch.Tensor
    ) -> torch.Tensor:
        """
        Content-based addressing using cosine similarity.

        Args:
            memory: Memory matrix (batch, N, W)
            key: Key vector (batch, W)
            strength: Strength scalar (batch, 1)

        Returns:
            Content weights (batch, N)
        """
        # Cosine similarity
        key = key.unsqueeze(1)  # (batch, 1, W)
        similarity = F.cosine_similarity(memory, key, dim=2)  # (batch, N)

        # Apply strength
        weights: torch.Tensor = F.softmax(strength * similarity, dim=1)
        return weights

    def update_memory(
        self,
        memory: torch.Tensor,
        write_weights: torch.Tensor,
        erase_vector: torch.Tensor,
        write_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update memory using write weights, erase vector, and write vector.

        Memory update: M_t = M_{t-1} * (1 - w_t^w * e_t^T) + w_t^w * a_t^T

        Args:
            memory: Current memory (batch, N, W)
            write_weights: Write weights (batch, N)
            erase_vector: Erase vector (batch, W)
            write_vector: Write vector (batch, W)

        Returns:
            Updated memory (batch, N, W)
        """
        # Erase
        erase = torch.einsum("bn,bw->bnw", write_weights, erase_vector)
        memory = memory * (1 - erase)

        # Write
        write = torch.einsum("bn,bw->bnw", write_weights, write_vector)
        memory = memory + write

        return memory

    def read_memory(
        self, memory: torch.Tensor, read_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Read from memory using read weights.

        Args:
            memory: Memory matrix (batch, N, W)
            read_weights: Read weights (batch, num_reads, N)

        Returns:
            Read vectors (batch, num_reads, W)
        """
        # read_vectors = sum_i w_i * M_i
        read_vectors: torch.Tensor = torch.einsum("brn,bnw->brw", read_weights, memory)
        return read_vectors


class DNC(nn.Module):
    """
    Differentiable Neural Computer (DNC).

    A neural network architecture with an external memory matrix
    that can be read from and written to in a differentiable manner.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension of controller network
        memory_size: Number of memory slots
        memory_dim: Dimension of each memory slot
        num_reads: Number of read heads
        output_dim: Output dimension
        controller_type: Type of controller ('lstm' or 'linear')
        output_type: 'prediction' returns output, 'embedding' returns controller state
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        hidden_dim: int = 128,
        memory_size: int = 64,
        memory_dim: int = 32,
        num_reads: int = 4,
        output_dim: int = 10,
        controller_type: Literal["lstm", "linear"] = "lstm",
        output_type: Literal["prediction", "embedding"] = "prediction",
    ) -> None:
        """Initialize DNC."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_reads = num_reads
        self.output_dim = output_dim
        self.output_type = output_type
        self.controller_type = controller_type

        # Controller network (LSTM or Linear)
        controller_input_dim = (
            input_dim + num_reads * memory_dim
        )  # Input + read vectors

        self.controller: nn.Module
        if controller_type == "lstm":
            self.controller = nn.LSTM(
                controller_input_dim, hidden_dim, batch_first=True
            )
        else:
            self.controller = nn.Linear(controller_input_dim, hidden_dim)

        # Memory module
        self.memory_module = DNCMemory(memory_size, memory_dim)

        # Interface parameters (generated from controller output)
        interface_size = (
            num_reads * memory_dim  # Read keys
            + num_reads  # Read strengths
            + memory_dim  # Write key
            + 1  # Write strength
            + memory_dim  # Erase vector
            + memory_dim  # Write vector
            + num_reads * 3  # Read modes (backward, content, forward)
        )

        self.interface_net = nn.Linear(hidden_dim, interface_size)

        # Output network
        self.output_net = nn.Linear(hidden_dim + num_reads * memory_dim, output_dim)

    def parse_interface_vector(
        self, interface: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Parse the interface vector into DNC parameters.

        Args:
            interface: Interface vector from controller (batch, interface_size) or (batch, seq, interface_size)

        Returns:
            Dictionary of parsed parameters
        """
        # interface usually (batch, hidden_dim) here because it's called per timestep
        batch_size = interface.size(0)
        idx = 0

        # Read keys (num_reads, memory_dim)
        read_keys = interface[:, idx : idx + self.num_reads * self.memory_dim]
        read_keys = read_keys.view(batch_size, self.num_reads, self.memory_dim)
        idx += self.num_reads * self.memory_dim

        # Read strengths (num_reads,)
        read_strengths = F.softplus(interface[:, idx : idx + self.num_reads]) + 1
        read_strengths = read_strengths.unsqueeze(-1)  # (batch, num_reads, 1)
        idx += self.num_reads

        # Write key (memory_dim,)
        write_key = interface[:, idx : idx + self.memory_dim]
        idx += self.memory_dim

        # Write strength (1,)
        write_strength = F.softplus(interface[:, idx : idx + 1]) + 1
        idx += 1

        # Erase vector (memory_dim,)
        erase_vector = torch.sigmoid(interface[:, idx : idx + self.memory_dim])
        idx += self.memory_dim

        # Write vector (memory_dim,)
        write_vector = interface[:, idx : idx + self.memory_dim]
        idx += self.memory_dim

        # Read modes (num_reads, 3) - backward, content, forward
        read_modes = interface[:, idx : idx + self.num_reads * 3]
        read_modes = read_modes.view(batch_size, self.num_reads, 3)
        read_modes = F.softmax(read_modes, dim=-1)

        return {
            "read_keys": read_keys,
            "read_strengths": read_strengths,
            "write_key": write_key,
            "write_strength": write_strength,
            "erase_vector": erase_vector,
            "write_vector": write_vector,
            "read_modes": read_modes,
        }

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        Forward pass through DNC.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            return_sequence: If True, return output for each timestep

        Returns:
            Output tensor of shape (batch_size, output_dim) or (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Initialize memory and states
        memory = torch.zeros(
            batch_size, self.memory_size, self.memory_dim, device=x.device
        )
        read_vectors = torch.zeros(
            batch_size, self.num_reads, self.memory_dim, device=x.device
        )

        h_state: torch.Tensor | None = None
        c_state: torch.Tensor | None = None
        if self.controller_type == "lstm":
            h_state = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)
            c_state = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)

        outputs: list[torch.Tensor] = []

        for t in range(seq_len):
            # Concatenate input with previous read vectors
            controller_input = torch.cat(
                [x[:, t, :], read_vectors.view(batch_size, -1)], dim=-1
            )

            # Controller forward pass
            if self.controller_type == "lstm":
                controller_input = controller_input.unsqueeze(1)  # Add time dimension
                controller_output, states = cast(nn.LSTM, self.controller)(
                    controller_input, (h_state, c_state)
                )
                h_state, c_state = states
                controller_output = controller_output.squeeze(1)
            else:
                controller_output = self.controller(controller_input)
                controller_output = torch.relu(controller_output)

            # Generate interface parameters
            interface = self.interface_net(controller_output)
            params = self.parse_interface_vector(interface)

            # Write to memory
            write_weights = self.memory_module.content_addressing(
                memory, params["write_key"], params["write_strength"]
            )
            memory = self.memory_module.update_memory(
                memory, write_weights, params["erase_vector"], params["write_vector"]
            )

            # Read from memory (simplified: content-based only)
            all_read_weights: list[torch.Tensor] = []
            for i in range(self.num_reads):
                weights = self.memory_module.content_addressing(
                    memory,
                    params["read_keys"][:, i, :],
                    params["read_strengths"][:, i, :],
                )
                all_read_weights.append(weights)

            read_weights = torch.stack(all_read_weights, dim=1)  # (batch, num_reads, N)

            # Get read vectors
            read_vectors = self.memory_module.read_memory(memory, read_weights)

            # Generate output
            output_input = torch.cat(
                [controller_output, read_vectors.view(batch_size, -1)], dim=-1
            )

            output = self.output_net(output_input)
            outputs.append(output)

        stacked_outputs = torch.stack(outputs, dim=1)

        if self.output_type == "embedding":
            if return_sequence:
                return stacked_outputs
            else:
                return stacked_outputs[:, -1, :]

        if return_sequence:
            return stacked_outputs
        else:
            return stacked_outputs[:, -1, :]
