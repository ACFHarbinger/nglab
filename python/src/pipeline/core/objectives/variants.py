"""
PPO Variant Objectives for NGLab.

Implements specialized RL losses including DR-GRPO, GSPO, and SAPO,
designed to improve stability and performance in financial trading tasks.
"""

import torch
from tensordict import TensorDict
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.utils import distance_loss


class DRGRPOLoss(ClipPPOLoss):
    """
    DR-GRPO: Group Relative Policy Optimization (Done Right).
    Features:
    1. Unnormalized centered advantages: A = R - Mean(R_group)
    2. No sequence length normalization in ratio (if applicable, though standard PPO ratio is per-step).

    In this implementation, we assume the 'advantage' key in the input TensorDict
    has already been computed via a group-based normalization if handled in the collector/buffer,
    OR we compute it here if 'reward' acts as the signal.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize DR-GRPO loss.
        """
        super().__init__(*args, **kwargs)
        # DR-GRPO specifically avoids certain normalizations if they were default
        # But here we assume the Advantage Estimation (GAE or Group) happens outside.
        pass

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """
        Forward pass for DR-GRPO.
        """
        # We can override forward to ensure specific advantage usage if needed.
        # For now, relying on standard PPO logic but assuming 'advantage' input
        # is Group-Relative: (R - Mean(R_group))
        return super().forward(tensordict)


class GSPOLoss(ClipPPOLoss):
    """
    GSPO: Group Sequence Policy Optimization.
    Key variation: Ratio is scaled by sequence length.
    ratio = exp( (log_new - log_old) / seq_len )
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize GSPO loss.
        """
        super().__init__(*args, **kwargs)

    def _log_ratio(self, tensordict):
        """
        Compute log ratio scaled by sequence length.
        """
        # We need to compute log_ratio = new_log_prob - old_log_prob
        # And then scale by sequence length.
        pass

    # Re-implementing forward for GSPO specifics is safer than inheriting partial logic
    # Placeholder: GSPO effectively modifies the 'ratio' variable.
    # We will implement a simplified version masking the standard one.
    pass


class SAPOLoss(ClipPPOLoss):
    """
    SAPO: Self-Adaptive Policy Optimization.
    Replaces clipping with a soft gating function.
    f(r) = (4/tau) * sigmoid(tau * (r - 1))
    """

    def __init__(self, tau_pos=0.1, tau_neg=0.5, *args, **kwargs):
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

        # Entropy and Critic Loss (Reuse or Re-compute)
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
                    else torch.tensor(0.0)
                ),
            },
            batch_size=[],
        )
