"""
Markov Chain (MC) implementation.
"""

import torch
from torch import nn


class MarkovChain(nn.Module):
    """
    Markov Chain (MC) - Probabilistic state transition model.
    Supports learnable and manual transition matrices.
    Can generate sequences and sample next states.
    """

    def __init__(
        self, num_states: int, output_type: str = "prediction", learnable: bool = True
    ) -> None:
        """Initialize Markov Chain."""
        super().__init__()
        self.num_states = num_states
        self.output_type = output_type

        # Transition matrix (unnormalized logits)
        if learnable:
            self.transition_matrix = nn.Parameter(torch.randn(num_states, num_states))
        else:
            self.register_buffer("transition_matrix", torch.eye(num_states))

    def get_transition_probs(self) -> torch.Tensor:
        """Get transition probabilities (softmax of logits)."""
        return torch.softmax(self.transition_matrix, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        x can be a sequence of states (discrete) or state probabilities.
        We assume x is (Batch, Seq, Num_States) or (Batch, Num_States).
        """
        probs = self.get_transition_probs()

        if x.dim() == 3:
            # Shift by one step using transition matrix
            # next_state_probs = current_state_probs @ T
            res = torch.matmul(x, probs)
        else:
            res = torch.matmul(x, probs)

        if not return_sequence and res.dim() == 3:
            return res[:, -1, :]
        return res

    def sample_next(self, current_state_idx: int) -> torch.Tensor:
        """Sample next state given current state index."""
        probs = self.get_transition_probs()[current_state_idx]
        return torch.multinomial(probs, 1)
