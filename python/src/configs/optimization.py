"""
Optimization Configurations.

Contains configurations for mixed precision training and profiling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

__all__ = ["MixedPrecisionConfig", "ProfilerConfig"]


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""

    # Precision mode
    precision: str = "16-mixed"  # 32, 16-mixed, bf16-mixed, 16-true, bf16-true

    # GradScaler settings (for FP16)
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    enabled: bool = True

    # Loss scaling
    dynamic_loss_scaling: bool = True
    static_loss_scale: float = 1.0

    # Optimization
    opt_level: str = "O1"  # O0, O1, O2, O3 (if using apex)

    @classmethod
    def from_env(cls) -> "MixedPrecisionConfig":
        """Create config from environment variables."""
        precision = os.getenv("NGLAB_PRECISION", "16-mixed")
        enabled = os.getenv("NGLAB_MIXED_PRECISION", "true").lower() == "true"
        return cls(precision=precision, enabled=enabled)

    @property
    def use_amp(self) -> bool:
        """Whether to use automatic mixed precision."""
        return self.precision in ("16-mixed", "bf16-mixed") and self.enabled

    @property
    def dtype(self) -> torch.dtype:
        """Get the target dtype for mixed precision."""
        if "bf16" in self.precision:
            return torch.bfloat16
        elif "16" in self.precision:
            return torch.float16
        return torch.float32


@dataclass
class ProfilerConfig:
    """Configuration for CUDA profiling."""

    # Output settings
    output_dir: str = "./profiler_output"
    export_chrome_trace: bool = True
    export_tensorboard: bool = True

    # Profiler schedule
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 3
    repeat: int = 1

    # Activities to profile
    profile_cpu: bool = True
    profile_cuda: bool = True

    # Memory profiling
    profile_memory: bool = True
    with_stack: bool = True
    with_flops: bool = True
    with_modules: bool = True

    # Record shapes for tensor operations
    record_shapes: bool = True
