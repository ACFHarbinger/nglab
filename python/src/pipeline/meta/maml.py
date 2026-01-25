"""
MAML Lightning Module for Meta-Learning Trading Strategies.

Integrates Model-Agnostic Meta-Learning with PyTorch Lightning training loop.
"""

import copy
from collections.abc import Iterator

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Optimizer


class MAMLLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for MAML (Model-Agnostic Meta-Learning).

    Enables meta-training across different market regimes with automatic
    support for distributed training, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        meta_batch_size: int = 4,
    ) -> None:
        """
        Initialize MAML Lightning module.

        Args:
            model: Base model to meta-train.
            inner_lr: Learning rate for inner loop (task adaptation).
            outer_lr: Learning rate for outer loop (meta-update).
            inner_steps: Number of gradient steps for adaptation.
            meta_batch_size: Number of tasks per meta-batch.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.meta_batch_size = meta_batch_size

    def configure_optimizers(self) -> Optimizer:
        """Configure meta-optimizer."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
        return optimizer

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
            support_x: Support set inputs [batch, seq_len, features].
            support_y: Support set targets [batch, output_dim].
            query_x: Query set inputs.
            query_y: Query set targets.

        Returns:
            Loss on query set after adaptation.
        """
        # Clone model for inner loop
        temp_model = copy.deepcopy(self.model)
        temp_optimizer = torch.optim.SGD(temp_model.parameters(), lr=self.inner_lr)

        # Adapt on support set
        temp_model.train()
        for _ in range(self.inner_steps):
            support_pred = temp_model(support_x)
            support_loss = torch.nn.functional.mse_loss(support_pred, support_y)
            temp_optimizer.zero_grad()
            support_loss.backward()
            temp_optimizer.step()

        # Evaluate on query set (keep gradients for meta-loss)
        query_pred = temp_model(query_x)
        query_loss = torch.nn.functional.mse_loss(query_pred, query_y)

        return query_loss

    def training_step(
        self,
        batch: dict[
            str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Meta-training step across multiple tasks.

        Args:
            batch: Dictionary containing 'tasks' list, where each task is
                   (support_x, support_y, query_x, query_y).
            batch_idx: Batch index.

        Returns:
            Meta-loss (average query loss across tasks).
        """
        tasks = batch["tasks"]
        meta_losses = []

        for support_x, support_y, query_x, query_y in tasks:
            query_loss = self.inner_loop(support_x, support_y, query_x, query_y)
            meta_losses.append(query_loss)

        # Meta-loss is average across tasks
        meta_loss = torch.stack(meta_losses).mean()

        # Log metrics
        self.log("train/meta_loss", meta_loss, prog_bar=True)
        self.log("train/num_tasks", len(tasks))

        return meta_loss

    def validation_step(
        self,
        batch: dict[
            str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Validation step to evaluate meta-learning performance.

        Args:
            batch: Dictionary containing validation tasks.
            batch_idx: Batch index.
        """
        tasks = batch["tasks"]
        val_losses = []

        # Validation doesn't need gradients through inner loop
        with torch.enable_grad():  # Re-enable for inner loop even in eval mode
            for support_x, support_y, query_x, query_y in tasks:
                query_loss = self.inner_loop(support_x, support_y, query_x, query_y)
                val_losses.append(query_loss.detach())  # Detach to avoid grad issues

        val_loss = torch.stack(val_losses).mean()
        self.log("val/meta_loss", val_loss, prog_bar=True)

        return val_loss

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
        steps = num_steps or self.inner_steps
        adapted_model = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        adapted_model.train()
        for _ in range(steps):
            pred = adapted_model(support_x)
            loss = torch.nn.functional.mse_loss(pred, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        adapted_model.eval()
        return adapted_model


class MAMLDataModule(pl.LightningDataModule):
    """
    DataModule for MAML meta-learning.

    Organizes data into tasks (market regimes) for meta-training.
    """

    def __init__(
        self,
        regime_datasets: dict[int, torch.Tensor],
        support_size: int = 50,
        query_size: int = 50,
        meta_batch_size: int = 4,
        num_workers: int = 0,
    ) -> None:
        """
        Initialize MAML DataModule.

        Args:
            regime_datasets: Dict mapping regime_id to dataset tensors.
            support_size: Number of samples for support set.
            query_size: Number of samples for query set.
            meta_batch_size: Number of tasks per meta-batch.
            num_workers: Number of workers for data loading.
        """
        super().__init__()
        self.regime_datasets = regime_datasets
        self.support_size = support_size
        self.query_size = query_size
        self.meta_batch_size = meta_batch_size
        self.num_workers = num_workers

    def create_task(
        self, regime_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create a task (support/query split) from regime data.

        Args:
            regime_data: Data from a specific regime [num_samples, features].

        Returns:
            (support_x, support_y, query_x, query_y)
        """
        total_size = self.support_size + self.query_size

        # Random sample from regime
        if regime_data.size(0) < total_size:
            # Repeat if not enough data
            indices = torch.randint(0, regime_data.size(0), (total_size,))
        else:
            indices = torch.randperm(regime_data.size(0))[:total_size]

        sampled = regime_data[indices]

        # Split into support/query
        support = sampled[: self.support_size]
        query = sampled[self.support_size : total_size]

        # For sequence data: [batch, seq_len, features]
        #  Model outputs predictions for last timestep only
        if support.dim() == 3:
            # Use last timestep features + target
            support_x = support[:, :, :-1]  # All timesteps, all features except last
            support_y = support[:, -1, -1:]  # Last timestep, last feature (target)
            query_x = query[:, :, :-1]
            query_y = query[:, -1, -1:]
        else:
            # 2D data - add sequence dimension
            support_x = support[:, :-1].unsqueeze(1)  # [batch, 1, features]
            support_y = support[:, -1:]  # [batch, 1]
            query_x = query[:, :-1].unsqueeze(1)
            query_y = query[:, -1:]

        return support_x, support_y, query_x, query_y

    def train_dataloader(
        self,
    ) -> Iterator[
        dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]
    ]:
        """Create training dataloader with task batches."""
        # For simplicity, create a static list of task batches
        # In practice, you'd implement a proper Dataset/DataLoader
        task_batches = []

        for _ in range(100):  # 100 meta-batches
            batch = []
            for _ in range(self.meta_batch_size):
                # Sample random regime
                regime_id = int(
                    torch.randint(0, len(self.regime_datasets), (1,)).item()
                )
                regime_data = self.regime_datasets[regime_id]
                task = self.create_task(regime_data)
                batch.append(task)
            task_batches.append({"tasks": batch})

        return iter(task_batches)

    def val_dataloader(
        self,
    ) -> Iterator[
        dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]
    ]:
        """Create validation dataloader."""
        # Similar to train but with fewer batches
        task_batches = []

        for _ in range(20):  # 20 validation batches
            batch = []
            for _ in range(self.meta_batch_size):
                regime_id = int(
                    torch.randint(0, len(self.regime_datasets), (1,)).item()
                )
                regime_data = self.regime_datasets[regime_id]
                task = self.create_task(regime_data)
                batch.append(task)
            task_batches.append({"tasks": batch})

        # Return as iterator to avoid multiple dataloader detection
        return iter(task_batches)
