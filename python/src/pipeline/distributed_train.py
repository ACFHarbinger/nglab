"""Distributed training utilities for NGLab."""

import logging
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_distributed() -> int:
    """
    Initialize the distributed process group and set the device.
    Returns:
        local_rank: The rank of the current process on the local node.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logger.warning(
            "Distributed environment variables not set. Defaulting to single-process."
        )
        return 0

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_distributed() -> None:
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(model: torch.nn.Module, local_rank: int) -> DDP:
    """
    Wrap a model with DistributedDataParallel.
    """
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)


class DistributedTrainer:
    """
    A boilerplate trainer for distributed training.
    """

    def __init__(  # noqa: PLR0913
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader[Any],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        local_rank: int,
        device: torch.device,
    ):
        """
        Initialize DistributedTrainer.

        Args:
            model: Model to train.
            train_loader: DataLoader with DistributedSampler.
            optimizer: Optimizer.
            criterion: Loss function.
            local_rank: Local GPU rank.
            device: Target device.
        """
        self.local_rank = local_rank
        self.device = device

        self.model = wrap_model_ddp(model, local_rank)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader

    def train_epoch(self, epoch: int) -> float:
        """Run a single training epoch."""
        self.model.train()
        sampler = self.train_loader.sampler
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        total_loss = 0.0
        # Use tqdm only on Rank 0
        loader = (
            tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=self.local_rank != 0)
            if self.local_rank == 0
            else self.train_loader
        )

        for _batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)  # noqa: PLW2901

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)


def train_ddp(  # noqa: PLR0913
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset[Any],
    optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
    criterion: torch.nn.Module,
    batch_size: int,
    n_epochs: int,
) -> None:
    """
    Standalone function to run DDP training.
    """
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    sampler: DistributedSampler[Any] = DistributedSampler(dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
    )

    trainer = DistributedTrainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer_factory(model),
        criterion=criterion,
        local_rank=local_rank,
        device=device,
    )

    for epoch in range(n_epochs):
        avg_loss = trainer.train_epoch(epoch)
        if local_rank == 0:
            logger.info(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

    cleanup_distributed()
