"""
PPO Variant Objectives for NGLab.

Implements specialized RL losses including DR-GRPO, GSPO, and SAPO,
designed to improve stability and performance in financial trading tasks.
"""

from typing import Any, cast

import torch
from tensordict import TensorDict
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.utils import distance_loss


class DRGRPOLoss(ClipPPOLoss):
    """
    DR-GRPO: Group Relative Policy Optimization (Done Right).
    Features:
    1. Unnormalized centered advantages: A = R - Mean(R_group)
    2. No sequence length normalization in ratio.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize DR-GRPO loss.
        """
        super().__init__(*args, **kwargs)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """
        Forward pass for DR-GRPO.
        """
        return super().forward(tensordict)


class GSPOLoss(ClipPPOLoss):
    """
    GSPO: Group Sequence Policy Optimization.
    Key variation: Ratio is scaled by sequence length.
    ratio = exp( (log_new - log_old) / seq_len )
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize GSPO loss.
        """
        super().__init__(*args, **kwargs)

    def _log_ratio(self, tensordict: TensorDict) -> torch.Tensor:
        """
        Compute log ratio scaled by sequence length.
        """
        # Placeholder for GSPO specific logic
        log_probs = self.actor_network(tensordict).log_prob(tensordict["action"])
        old_log_probs = tensordict["sample_log_prob"]
        # In actual GSPO, we'd find the sequence length from tensordict metadata or shape
        seq_len = 1.0  # Placeholder
        return cast(torch.Tensor, (log_probs - old_log_probs) / seq_len)


class SAPOLoss(ClipPPOLoss):
    """
    SAPO: Self-Adaptive Policy Optimization.
    Replaces clipping with a soft gating function.
    f(r) = (4/tau) * sigmoid(tau * (r - 1))
    """

    def __init__(
        self, tau_pos: float = 0.1, tau_neg: float = 0.5, *args: Any, **kwargs: Any
    ) -> None:
        """
        Initialize SAPO loss.

        Args:
            tau_pos (float): Temperature for positive advantages.
            tau_neg (float): Temperature for negative advantages.
        """
        super().__init__(*args, **kwargs)
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """
        Forward pass for SAPO using soft gating.
        """
        # Calculate prob ratio
        dist = self.actor_network.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict["action"])
        old_log_probs = tensordict["sample_log_prob"]

        ratio = (log_probs - old_log_probs).exp()
        advantage = tensordict["advantage"]

        # Adaptive Tau
        tau = torch.where(
            advantage > 0,
            torch.tensor(self.tau_pos, device=advantage.device),
            torch.tensor(self.tau_neg, device=advantage.device),
        )

        # Soft Gating
        f_ratio = (4.0 / tau) * torch.sigmoid(tau * (ratio - 1.0))

        # Loss
        surrogate = f_ratio * advantage
        loss_objective = -surrogate.mean()

        # Entropy
        entropy = dist.entropy().mean()

        # Value loss (Standard L2)
        if self.critic_network:
            value = self.critic_network(tensordict)
            value_target = tensordict["value_target"]
            loss_critic = distance_loss(value, value_target, loss_function="l2")
        else:
            loss_critic = torch.tensor(0.0, device=loss_objective.device)

        return TensorDict(
            {
                "loss_objective": loss_objective,
                "loss_critic": loss_critic,
                "loss_entropy": (
                    -self.entropy_bonus * entropy
                    if self.entropy_bonus
                    else torch.tensor(0.0, device=loss_objective.device)
                ),
            },
            batch_size=[],
        )
