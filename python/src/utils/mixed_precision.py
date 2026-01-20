"""
Mixed Precision Training Utilities for NGLab.

Provides utilities for FP16/BF16 mixed precision training to improve
GPU utilization and reduce memory usage while maintaining model accuracy.
"""

import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast


class PrecisionMode(Enum):
    """Supported precision modes."""

    FP32 = "32"
    FP16_MIXED = "16-mixed"
    BF16_MIXED = "bf16-mixed"
    FP16_TRUE = "16-true"
    BF16_TRUE = "bf16-true"


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


class MixedPrecisionTrainer:
    """
    Mixed precision training wrapper.

    Handles automatic mixed precision (AMP) training with gradient scaling
    for FP16 to prevent underflow.

    Example:
        config = MixedPrecisionConfig(precision="16-mixed")
        trainer = MixedPrecisionTrainer(model, optimizer, config)

        for batch in dataloader:
            loss = trainer.training_step(batch, forward_fn, loss_fn)
            trainer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: MixedPrecisionConfig | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or MixedPrecisionConfig()

        # Initialize GradScaler for FP16 mixed precision
        self._scaler: GradScaler | None = None
        if self.config.use_amp and self.config.precision == "16-mixed":
            self._scaler = GradScaler(
                init_scale=self.config.init_scale,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval,
                enabled=True,
            )

        # Device type for autocast
        self._device_type = self._get_device_type()

    def _get_device_type(self) -> str:
        """Get device type for autocast context."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision."""
        if self.config.use_amp:
            with autocast(device_type=self._device_type, dtype=self.config.dtype):
                yield
        else:
            yield

    def training_step(
        self,
        batch: Any,
        forward_fn: Callable[[nn.Module, Any], torch.Tensor],
        loss_fn: Callable[[torch.Tensor, Any], torch.Tensor],
        accumulation_steps: int = 1,
    ) -> tuple[torch.Tensor, float]:
        """
        Execute a single training step with mixed precision.

        Args:
            batch: Input batch
            forward_fn: Function that takes (model, batch) and returns output
            loss_fn: Function that takes (output, batch) and returns loss
            accumulation_steps: Gradient accumulation steps

        Returns:
            Tuple of (loss tensor, loss value)
        """
        self.optimizer.zero_grad()

        with self.autocast_context():
            output = forward_fn(self.model, batch)
            loss = loss_fn(output, batch)
            loss = loss / accumulation_steps

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss, loss.item() * accumulation_steps

    def step(self, clip_grad_norm: float | None = None) -> None:
        """
        Execute optimizer step with gradient unscaling and optional clipping.

        Args:
            clip_grad_norm: Maximum gradient norm for clipping (None to disable)
        """
        if self._scaler is not None:
            # Unscale gradients for clipping
            if clip_grad_norm is not None:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)

            # Step with scaler
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            # Standard step
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
            self.optimizer.step()

    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        state = {
            "config": {
                "precision": self.config.precision,
                "enabled": self.config.enabled,
            }
        }
        if self._scaler is not None:
            state["scaler"] = self._scaler.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        """Load state dict from checkpoint."""
        if self._scaler is not None and "scaler" in state:
            self._scaler.load_state_dict(state["scaler"])

    def get_scale(self) -> float:
        """Get current loss scale value."""
        if self._scaler is not None:
            return self._scaler.get_scale()
        return 1.0


def configure_model_for_mixed_precision(
    model: nn.Module,
    precision: str = "16-mixed",
) -> nn.Module:
    """
    Configure model layers for optimal mixed precision performance.

    Some operations are better kept in FP32 for numerical stability:
    - Layer normalization
    - Softmax
    - Loss computation

    Args:
        model: PyTorch model
        precision: Precision mode

    Returns:
        Configured model
    """
    if precision == "32":
        return model

    # These layers should stay in FP32 for stability
    fp32_layers = (
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
    )

    for module in model.modules():
        if isinstance(module, fp32_layers):
            # Keep normalization layers in FP32
            module.float()

    return model


def get_optimal_precision() -> str:
    """
    Detect and return the optimal precision mode for the current hardware.

    Returns:
        Optimal precision string (e.g., "16-mixed", "bf16-mixed", "32")
    """
    if not torch.cuda.is_available():
        # CPU: BF16 might be available on newer CPUs
        if hasattr(torch, "bfloat16"):
            try:
                # Test if BF16 operations work
                x = torch.randn(10, dtype=torch.bfloat16)
                _ = x + x
                return "bf16-mixed"
            except Exception:
                pass
        return "32"

    # Get GPU compute capability
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    major, _minor = capability

    # Ampere (SM 8.0+) and newer: BF16 is preferred
    if major >= 8:
        return "bf16-mixed"

    # Volta (SM 7.0+) and Turing (SM 7.5): FP16 with tensor cores
    if major >= 7:
        return "16-mixed"

    # Older GPUs: FP32 is safer
    return "32"


def estimate_memory_savings(
    model: nn.Module,
    batch_size: int,
    sequence_length: int,
    precision: str = "16-mixed",
) -> dict:
    """
    Estimate memory savings from using mixed precision.

    Args:
        model: PyTorch model
        batch_size: Training batch size
        sequence_length: Input sequence length
        precision: Target precision mode

    Returns:
        Dictionary with memory estimates
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # FP32 bytes per parameter
    fp32_bytes = 4
    fp16_bytes = 2

    # Model weights memory
    fp32_model_memory = total_params * fp32_bytes / (1024**2)  # MB

    if "16" in precision or "bf16" in precision:
        mixed_model_memory = total_params * fp16_bytes / (1024**2)
    else:
        mixed_model_memory = fp32_model_memory

    # Optimizer states (Adam has 2 states per param)
    fp32_optimizer_memory = trainable_params * fp32_bytes * 2 / (1024**2)

    # Gradients
    fp32_gradient_memory = trainable_params * fp32_bytes / (1024**2)
    mixed_gradient_memory = trainable_params * fp16_bytes / (1024**2)

    # Activations (rough estimate based on batch size and sequence length)
    # This is highly model-dependent
    activation_factor = batch_size * sequence_length
    fp32_activation_memory = (
        activation_factor * total_params * 0.01 * fp32_bytes / (1024**2)
    )
    mixed_activation_memory = (
        activation_factor * total_params * 0.01 * fp16_bytes / (1024**2)
    )

    fp32_total = (
        fp32_model_memory
        + fp32_optimizer_memory
        + fp32_gradient_memory
        + fp32_activation_memory
    )

    mixed_total = (
        mixed_model_memory
        + fp32_optimizer_memory  # Optimizer stays FP32
        + mixed_gradient_memory
        + mixed_activation_memory
    )

    savings_mb = fp32_total - mixed_total
    savings_percent = (savings_mb / fp32_total) * 100 if fp32_total > 0 else 0

    return {
        "fp32_total_mb": fp32_total,
        "mixed_total_mb": mixed_total,
        "savings_mb": savings_mb,
        "savings_percent": savings_percent,
        "model_params": total_params,
        "trainable_params": trainable_params,
        "precision": precision,
    }
