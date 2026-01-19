import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)

def setup_distributed() -> int:
    """
    Initialize the distributed process group and set the device.
    Returns:
        local_rank: The rank of the current process on the local node.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logger.warning("Distributed environment variables not set. Defaulting to single-process.")
        return 0

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
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
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader_factory: Callable[[Optional[DistributedSampler[Any]]], torch.utils.data.DataLoader[Any]],
        optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
        criterion: torch.nn.Module,
        local_rank: int,
    ):
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        self.model = wrap_model_ddp(model, local_rank)
        self.optimizer = optimizer_factory(self.model)
        self.criterion = criterion
        
        # Sampler is required for DDP to ensure each process sees different data
        # Note: This factory pattern is a placeholder. 
        # In a real implementation, the factory should handle dataset/sampler internally.
        self.train_loader = train_loader_factory(None) 

    def train_epoch(self, epoch: int) -> float:
        self.model.train()  # type: ignore[no-untyped-call]
        # Set epoch for sampler to ensure different shuffling across epochs
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
            
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
