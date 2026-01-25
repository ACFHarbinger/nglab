"""
GPU Optimization Utilities for NGLab.

Provides memory pre-allocation, transfer profiling between Python↔Rust,
and utilities for identifying and resolving GPU bottlenecks.
"""

import gc
import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class TransferProfile:
    """Profile of a Python↔Rust data transfer operation."""

    name: str
    transfer_time_ms: float
    data_size_bytes: int
    direction: str  # "py_to_rust" or "rust_to_py"
    bandwidth_gbps: float = 0.0

    def __post_init__(self) -> None:
        """Calculate bandwidth after initialization."""
        if self.transfer_time_ms > 0:
            # Calculate bandwidth in GB/s
            bytes_per_ms = self.data_size_bytes / self.transfer_time_ms
            self.bandwidth_gbps = bytes_per_ms * 1000 / (1024**3)


@dataclass
class MemoryPool:
    """Pre-allocated GPU memory pool for reducing allocation overhead."""

    device: int
    pool_size_mb: float
    allocated_tensors: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        device: int = 0,
        pool_size_mb: float = 1024.0,
    ) -> "MemoryPool":
        """Create a memory pool by pre-allocating GPU memory.

        Args:
            device: CUDA device index.
            pool_size_mb: Pool size in megabytes.

        Returns:
            Initialized MemoryPool.
        """
        pool = cls(device=device, pool_size_mb=pool_size_mb)

        # Pre-allocate a large tensor to "warm up" the CUDA memory allocator
        elements = int((pool_size_mb * 1024 * 1024) / 4)  # 4 bytes per float32
        try:
            warmup_tensor = torch.empty(
                elements,
                dtype=torch.float32,
                device=f"cuda:{device}",
            )
            del warmup_tensor
            torch.cuda.empty_cache()
            logger.info(
                f"Pre-allocated {pool_size_mb}MB GPU memory pool on device {device}"
            )
        except RuntimeError as e:
            logger.warning(f"Failed to pre-allocate memory pool: {e}")

        return pool

    def allocate(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Allocate a named tensor from the pool.

        Args:
            name: Tensor name for later retrieval.
            shape: Tensor shape.
            dtype: Tensor dtype.

        Returns:
            Pre-allocated tensor.
        """
        if name in self.allocated_tensors:
            existing = self.allocated_tensors[name]
            if existing.shape == shape and existing.dtype == dtype:
                return existing
            # Shape mismatch, deallocate old tensor
            del self.allocated_tensors[name]

        tensor = torch.empty(
            shape,
            dtype=dtype,
            device=f"cuda:{self.device}",
        )
        self.allocated_tensors[name] = tensor
        return tensor

    def get(self, name: str) -> torch.Tensor | None:
        """Get a previously allocated tensor by name."""
        return self.allocated_tensors.get(name)

    def deallocate(self, name: str) -> None:
        """Deallocate a tensor from the pool."""
        if name in self.allocated_tensors:
            del self.allocated_tensors[name]

    def clear(self) -> None:
        """Clear all allocated tensors."""
        self.allocated_tensors.clear()
        torch.cuda.empty_cache()


class TransferProfiler:
    """Profiler for Python↔Rust data transfers.

    Measures the overhead of transferring data between Python (numpy/torch)
    and Rust (via PyO3 bindings).
    """

    def __init__(self) -> None:
        """Initialize transfer profiler."""
        self.profiles: list[TransferProfile] = []
        self._start_time: float = 0.0
        self._current_name: str = ""
        self._current_direction: str = ""

    @contextmanager
    def profile_transfer(
        self,
        name: str,
        direction: str = "py_to_rust",
        data_size_bytes: int | None = None,
    ) -> Generator[None, None, None]:
        """Context manager for profiling a data transfer.

        Args:
            name: Name of the transfer operation.
            direction: "py_to_rust" or "rust_to_py".
            data_size_bytes: Optional data size (will estimate if not provided).
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        yield

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Default data size if not provided
        size = data_size_bytes if data_size_bytes is not None else 0

        profile = TransferProfile(
            name=name,
            transfer_time_ms=elapsed_ms,
            data_size_bytes=size,
            direction=direction,
        )
        self.profiles.append(profile)

    def profile_array_transfer(
        self,
        name: str,
        array: Any,
        direction: str = "py_to_rust",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for profiling array transfers.

        Args:
            name: Name of the transfer.
            array: NumPy or torch array to measure size.
            direction: Transfer direction.

        Returns:
            Decorator function.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            """Inner decorator for measurement."""

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                """Wrapper that profiles array transfer."""
                # Get data size
                size = 0
                if hasattr(array, "nbytes"):
                    size = int(array.nbytes)
                elif hasattr(array, "element_size") and hasattr(array, "numel"):
                    size = int(array.element_size() * array.numel())

                with self.profile_transfer(name, direction, size):
                    result = func(*args, **kwargs)
                return result

            return wrapper

        return decorator

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of all profiles."""
        if not self.profiles:
            return {"total_transfers": 0}

        py_to_rust = [p for p in self.profiles if p.direction == "py_to_rust"]
        rust_to_py = [p for p in self.profiles if p.direction == "rust_to_py"]

        def stats(profiles: list[TransferProfile]) -> dict[str, float]:
            """Calculate statistics for a subset of profiles."""
            if not profiles:
                return {"count": 0.0, "total_ms": 0.0, "avg_ms": 0.0}
            times = [p.transfer_time_ms for p in profiles]
            return {
                "count": float(len(profiles)),
                "total_ms": float(sum(times)),
                "avg_ms": float(sum(times) / len(times)),
                "max_ms": float(max(times)),
                "min_ms": float(min(times)),
            }

        return {
            "total_transfers": len(self.profiles),
            "py_to_rust": stats(py_to_rust),
            "rust_to_py": stats(rust_to_py),
            "total_time_ms": sum(p.transfer_time_ms for p in self.profiles),
        }

    def print_summary(self) -> None:
        """Print profiling summary to console."""
        summary = self.get_summary()

        print("\n=== Transfer Profiling Summary ===")
        print(f"Total transfers: {summary['total_transfers']}")
        print(f"Total time: {summary['total_time_ms']:.2f}ms")

        for direction in ["py_to_rust", "rust_to_py"]:
            stats_dict = summary.get(direction, {})
            if stats_dict.get("count", 0.0) > 0.0:
                print(f"\n{direction}:")
                print(f"  Count: {int(stats_dict['count'])}")
                print(f"  Total: {stats_dict['total_ms']:.2f}ms")
                print(f"  Avg: {stats_dict['avg_ms']:.2f}ms")
                print(
                    f"  Min/Max: {stats_dict['min_ms']:.2f}ms / {stats_dict['max_ms']:.2f}ms"
                )

    def clear(self) -> None:
        """Clear all profiles."""
        self.profiles.clear()


class GPUMemoryOptimizer:
    """Utilities for optimizing GPU memory usage.

    Provides memory tracking, garbage collection, and optimization hints.
    """

    def __init__(self, device: int = 0) -> None:
        """Initialize TransferProfiler."""
        self.device = device
        self._snapshots: list[dict[str, Any]] = []

    def snapshot(self, label: str = "") -> dict[str, Any]:
        """Take a memory snapshot."""
        if not torch.cuda.is_available():
            return {"available": False}

        torch.cuda.synchronize()

        snapshot = {
            "label": label,
            "timestamp": time.time(),
            "allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1024**2,
        }

        self._snapshots.append(snapshot)
        return snapshot

    def compare_snapshots(
        self,
        before_label: str,
        after_label: str,
    ) -> dict[str, float]:
        """Compare two labeled snapshots."""
        before = next(
            (s for s in self._snapshots if s.get("label") == before_label), None
        )
        after = next(
            (s for s in self._snapshots if s.get("label") == after_label), None
        )

        if before is None or after is None:
            return {"error": -1.0}  # Signal error with negative value

        return {
            "allocated_diff_mb": float(after["allocated_mb"] - before["allocated_mb"]),
            "reserved_diff_mb": float(after["reserved_mb"] - before["reserved_mb"]),
            "time_diff_s": float(after["timestamp"] - before["timestamp"]),
        }

    @staticmethod
    def aggressive_cleanup() -> dict[str, float]:
        """Perform aggressive GPU memory cleanup."""
        if not torch.cuda.is_available():
            return {"available": 0.0}

        before = torch.cuda.memory_allocated() / 1024**2

        # Python garbage collection
        gc.collect()

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

        after = torch.cuda.memory_allocated() / 1024**2

        return {
            "before_mb": before,
            "after_mb": after,
            "freed_mb": before - after,
        }

    @staticmethod
    def get_memory_bottlenecks(model: nn.Module) -> list[dict[str, Any]]:
        """Identify potential memory bottlenecks in a model."""
        bottlenecks: list[dict[str, Any]] = []

        total_params = 0
        for name, param in model.named_parameters():
            param_size = param.numel() * param.element_size()
            total_params += param_size

            # Flag large individual parameters
            if param_size > 100 * 1024 * 1024:  # > 100MB
                bottlenecks.append(
                    {
                        "type": "large_parameter",
                        "name": name,
                        "size_mb": param_size / 1024**2,
                        "recommendation": "Consider gradient checkpointing or parameter sharing",
                    }
                )

        # Check for modules that might benefit from optimization
        for name, module in model.named_modules():
            # Large embeddings
            if isinstance(module, nn.Embedding):
                embed_size = module.num_embeddings * module.embedding_dim * 4 / 1024**2
                if embed_size > 50:
                    bottlenecks.append(
                        {
                            "type": "large_embedding",
                            "name": name,
                            "size_mb": embed_size,
                            "recommendation": "Consider embedding compression or sparse embeddings",
                        }
                    )

            # Attention layers without efficient attention
            if "Attention" in module.__class__.__name__:
                bottlenecks.append(
                    {
                        "type": "attention",
                        "name": name,
                        "recommendation": "Consider Flash Attention or memory-efficient attention",
                    }
                )

        return bottlenecks

    @staticmethod
    def estimate_batch_memory(
        model: nn.Module,
        input_shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        with_grad: bool = True,
    ) -> dict[str, float]:
        """Estimate memory usage for a batch."""
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()

        # Model parameters
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        # Model gradients (same size as parameters)
        grad_bytes = param_bytes if with_grad else 0

        # Input tensor
        input_elements = 1
        for dim in input_shape:
            input_elements *= dim
        input_bytes = input_elements * bytes_per_element

        # Rough estimate of activations (typically 2-4x parameters for transformers)
        activation_bytes = param_bytes * 3 if with_grad else param_bytes

        # Optimizer states (Adam uses 2 additional buffers per parameter)
        optimizer_bytes = param_bytes * 2 if with_grad else 0

        total_bytes = (
            param_bytes + grad_bytes + input_bytes + activation_bytes + optimizer_bytes
        )

        return {
            "parameters_mb": param_bytes / 1024**2,
            "gradients_mb": grad_bytes / 1024**2,
            "input_mb": input_bytes / 1024**2,
            "activations_mb": activation_bytes / 1024**2,
            "optimizer_mb": optimizer_bytes / 1024**2,
            "total_estimated_mb": total_bytes / 1024**2,
        }


def enable_memory_efficient_attention() -> bool:
    """Enable memory-efficient attention if available."""
    try:
        # Check for PyTorch 2.0+ scaled_dot_product_attention
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Enable flash attention if available
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            logger.info("Enabled memory-efficient attention (SDPA)")
            return True
    except Exception as e:
        logger.warning(f"Could not enable memory-efficient attention: {e}")
    return False


def optimize_for_inference(model: nn.Module) -> nn.Module:
    """Optimize model for inference."""
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    # Try torch.compile (PyTorch 2.0+)
    try:
        if hasattr(torch, "compile"):
            compiled = torch.compile(model, mode="reduce-overhead")
            if isinstance(compiled, nn.Module):
                model = compiled
            logger.info("Applied torch.compile optimization")
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}")

    return model


def get_gpu_optimization_recommendations() -> list[str]:
    """Get GPU optimization recommendations based on current setup."""
    recommendations = []

    if not torch.cuda.is_available():
        recommendations.append("CUDA not available - consider using GPU for training")
        return recommendations

    # Check GPU compute capability
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)

    if capability[0] >= 8:  # Ampere or newer
        recommendations.append(
            "✅ GPU supports BF16 - use bf16-mixed precision for best performance"
        )
        recommendations.append(
            "✅ GPU supports Flash Attention - enable via enable_memory_efficient_attention()"
        )
    elif capability[0] >= 7:  # Volta/Turing
        recommendations.append(
            "✅ GPU supports FP16 Tensor Cores - use 16-mixed precision"
        )
    else:
        recommendations.append(
            "⚠️ Older GPU - mixed precision may not provide significant speedup"
        )

    # Memory recommendations
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    if total_memory < 8:
        recommendations.append("⚠️ Limited GPU memory - use gradient checkpointing")
        recommendations.append(
            "⚠️ Consider smaller batch sizes or gradient accumulation"
        )

    # CUDA version
    cuda_version = torch.version.cuda
    if cuda_version:
        major = int(cuda_version.split(".")[0])
        if major < 11:
            recommendations.append("⚠️ CUDA < 11 - upgrade for better performance")

    return recommendations
