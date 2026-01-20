"""
MAML (Model-Agnostic Meta-Learning) Wrapper for Trading Strategies.

Enables rapid adaptation to changing market conditions with few gradient steps.
"""

import copy

import torch
from torch import nn
from torch.optim import Adam


class MAMLWrapper(nn.Module):
    """
    MAML wrapper for any PyTorch model.

    Trains model initialization for fast adaptation to new tasks (market regimes).
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
    ):
        """
        Initialize MAML wrapper.

        Args:
            model: Base model to wrap.
            inner_lr: Learning rate for inner loop (adaptation).
            outer_lr: Learning rate for outer loop (meta-update).
            inner_steps: Number of gradient steps for adaptation.
        """
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = Adam(self.model.parameters(), lr=outer_lr)

    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform inner loop adaptation on support set and evaluate on query set.

        Args:
            support_x: Support set inputs.
            support_y: Support set targets.
            query_x: Query set inputs.
            query_y: Query set targets.

        Returns:
            Loss on query set after adaptation.
        """
        # Clone model for inner loop
        temp_model = copy.deepcopy(self.model)
        temp_optimizer = Adam(temp_model.parameters(), lr=self.inner_lr)

        # Adapt on support set
        for _ in range(self.inner_steps):
            support_pred = temp_model(support_x)
            support_loss = nn.functional.mse_loss(support_pred, support_y)
            temp_optimizer.zero_grad()
            support_loss.backward()
            temp_optimizer.step()

        # Evaluate on query set
        query_pred = temp_model(query_x)
        query_loss = nn.functional.mse_loss(query_pred, query_y)

        return query_loss

    def meta_train_step(
        self,
        task_batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> float:
        """
        Perform one meta-training step across a batch of tasks.

        Args:
            task_batch: List of (support_x, support_y, query_x, query_y) tuples.

        Returns:
            Average meta-loss across tasks.
        """
        self.meta_optimizer.zero_grad()
        meta_losses = []

        for support_x, support_y, query_x, query_y in task_batch:
            query_loss = self.inner_loop(support_x, support_y, query_x, query_y)
            meta_losses.append(query_loss)

        # Meta-loss is average across tasks
        meta_loss = torch.stack(meta_losses).mean()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        num_steps: int | None = None,
    ) -> nn.Module:
        """
        Fast adaptation to new market regime.

        Args:
            support_x: Support set inputs from new regime.
            support_y: Support set targets from new regime.
            num_steps: Number of adaptation steps (defaults to self.inner_steps).

        Returns:
            Adapted model.
        """
        num_steps = num_steps or self.inner_steps
        adapted_model = copy.deepcopy(self.model)
        optimizer = Adam(adapted_model.parameters(), lr=self.inner_lr)

        for _ in range(num_steps):
            pred = adapted_model(support_x)
            loss = nn.functional.mse_loss(pred, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return adapted_model


def create_task_from_regime_data(
    regime_data: torch.Tensor, support_size: int = 50, query_size: int = 50
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split regime data into support and query sets for MAML.

    Args:
        regime_data: Data from a specific market regime [num_samples, seq_len, features].
        support_size: Number of samples for support set.
        query_size: Number of samples for query set.

    Returns:
        (support_x, support_y, query_x, query_y)
    """
    # Simple split for now - in practice would use more sophisticated splitting
    total_size = support_size + query_size
    assert regime_data.size(0) >= total_size, "Not enough data for task"

    support_x = regime_data[:support_size, :-1]  # All but last column as input
    support_y = regime_data[:support_size, -1:]  # Last column as target
    query_x = regime_data[support_size:total_size, :-1]
    query_y = regime_data[support_size:total_size, -1:]

    return support_x, support_y, query_x, query_y
