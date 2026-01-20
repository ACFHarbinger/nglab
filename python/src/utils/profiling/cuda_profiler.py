"""
CUDA Profiling utilities using torch.profiler.

Provides detailed GPU performance analysis including:
- CUDA kernel timing
- Memory allocation patterns
- Tensor operations analysis
- Chrome trace export for visualization
"""

import os
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import torch
from torch import nn
from torch.profiler import (
    ProfilerActivity,
    profile as torch_profile,
    schedule,
    tensorboard_trace_handler,
)


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


@dataclass
class GPUMemoryStats:
    """GPU memory statistics."""

    device: int
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float
    free_mb: float
    total_mb: float
    utilization_percent: float


@dataclass
class ProfilingResult:
    """Results from profiling session."""

    total_cuda_time_ms: float
    total_cpu_time_ms: float
    peak_memory_mb: float
    avg_memory_mb: float
    top_operations: List[Dict[str, Any]] = field(default_factory=list)
    chrome_trace_path: Optional[str] = None
    tensorboard_dir: Optional[str] = None


class CUDAProfiler:
    """
    CUDA profiler for detailed GPU performance analysis.

    Example usage:
        profiler = CUDAProfiler(config)
        with profiler.profile() as prof:
            for batch in dataloader:
                output = model(batch)
                prof.step()
        result = profiler.get_results()
    """

    def __init__(self, config: Optional[ProfilerConfig] = None) -> None:
        self.config = config or ProfilerConfig()
        self._profiler: Optional[torch_profile] = None
        self._step_count = 0
        self._results: Optional[ProfilingResult] = None

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    @contextmanager
    def profile(self) -> Generator["CUDAProfiler", None, None]:
        """Context manager for profiling."""
        activities = []
        if self.config.profile_cpu:
            activities.append(ProfilerActivity.CPU)
        if self.config.profile_cuda and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        # Setup schedule
        prof_schedule = schedule(
            wait=self.config.wait_steps,
            warmup=self.config.warmup_steps,
            active=self.config.active_steps,
            repeat=self.config.repeat,
        )

        # Setup trace handler
        trace_handler = None
        if self.config.export_tensorboard:
            tb_dir = os.path.join(
                self.config.output_dir,
                f"tensorboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            trace_handler = tensorboard_trace_handler(tb_dir)

        with torch_profile(
            activities=activities,
            schedule=prof_schedule,
            on_trace_ready=trace_handler,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            with_modules=self.config.with_modules,
        ) as prof:
            self._profiler = prof
            try:
                yield self
            finally:
                self._profiler = None

            # Export chrome trace
            if self.config.export_chrome_trace:
                trace_path = os.path.join(
                    self.config.output_dir,
                    f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                )
                prof.export_chrome_trace(trace_path)

            # Collect results
            self._collect_results(prof)

    def step(self) -> None:
        """Signal profiler to move to next step."""
        if self._profiler is not None:
            self._profiler.step()
            self._step_count += 1

    def _collect_results(self, prof: torch_profile) -> None:
        """Collect profiling results."""
        key_averages = prof.key_averages()

        # Calculate totals
        total_cuda_time = sum(
            float(event.cuda_time_total) for event in key_averages if event.cuda_time_total
        )
        total_cpu_time = sum(
            float(event.cpu_time_total) for event in key_averages if event.cpu_time_total
        )

        # Get memory stats
        if torch.cuda.is_available():
            peak_memory = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_memory = 0.0

        # Get top operations by CUDA time
        top_ops: List[Dict[str, Any]] = []
        for event in sorted(
            key_averages, key=lambda x: x.cuda_time_total or 0, reverse=True
        )[:10]:
            top_ops.append(
                {
                    "name": event.key,
                    "cuda_time_ms": (event.cuda_time_total or 0) / 1000,
                    "cpu_time_ms": (event.cpu_time_total or 0) / 1000,
                    "calls": event.count,
                    "memory_mb": (event.self_cpu_memory_usage or 0) / (1024 * 1024),
                }
            )

        self._results = ProfilingResult(
            total_cuda_time_ms=total_cuda_time / 1000,
            total_cpu_time_ms=total_cpu_time / 1000,
            peak_memory_mb=peak_memory,
            avg_memory_mb=peak_memory / 2,  # Approximation
            top_operations=top_ops,
        )

    def get_results(self) -> Optional[ProfilingResult]:
        """Get profiling results."""
        return self._results

    def print_summary(self) -> None:
        """Print profiling summary to console."""
        if self._results is None:
            print("No profiling results available.")
            return

        print("\n" + "=" * 60)
        print("CUDA PROFILING SUMMARY")
        print("=" * 60)
        print(f"Total CUDA time: {self._results.total_cuda_time_ms:.2f} ms")
        print(f"Total CPU time:  {self._results.total_cpu_time_ms:.2f} ms")
        print(f"Peak memory:     {self._results.peak_memory_mb:.2f} MB")
        print("\nTop 10 operations by CUDA time:")
        print("-" * 60)
        for i, op in enumerate(self._results.top_operations, 1):
            print(
                f"{i:2d}. {op['name'][:40]:<40} "
                f"CUDA: {op['cuda_time_ms']:>8.2f}ms "
                f"calls: {op['calls']:>5d}"
            )
        print("=" * 60 + "\n")


def get_gpu_memory_stats(device: int = 0) -> Optional[GPUMemoryStats]:
    """
    Get current GPU memory statistics.

    Args:
        device: CUDA device index

    Returns:
        GPUMemoryStats or None if CUDA unavailable
    """
    if not torch.cuda.is_available():
        return None

    torch.cuda.synchronize(device)

    allocated = float(torch.cuda.memory_allocated(device) / (1024 * 1024))
    reserved = float(torch.cuda.memory_reserved(device) / (1024 * 1024))
    max_allocated = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    max_reserved = float(torch.cuda.max_memory_reserved(device) / (1024 * 1024))

    # Get total memory from device properties
    props = torch.cuda.get_device_properties(device)
    total = float(props.total_memory / (1024 * 1024))
    free = total - reserved

    return GPUMemoryStats(
        device=device,
        allocated_mb=allocated,
        reserved_mb=reserved,
        max_allocated_mb=max_allocated,
        max_reserved_mb=max_reserved,
        free_mb=free,
        total_mb=total,
        utilization_percent=(allocated / total) * 100,
    )


def profile_model_forward(
    model: nn.Module,
    sample_input: torch.Tensor,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> Dict[str, Any]:
    """
    Profile model forward pass.

    Args:
        model: PyTorch model to profile
        sample_input: Sample input tensor
        num_iterations: Number of profiling iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with timing statistics
    """
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(sample_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Profile
    times: List[float] = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if torch.cuda.is_available():
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                _ = model(sample_input)
                end_evt.record()
                torch.cuda.synchronize()
                times.append(float(start_evt.elapsed_time(end_evt)))
            else:
                start_time = time.perf_counter()
                _ = model(sample_input)
                times.append(float((time.perf_counter() - start_time) * 1000))

    times_tensor = torch.tensor(times)
    return {
        "mean_ms": float(times_tensor.mean().item()),
        "std_ms": float(times_tensor.std().item()),
        "min_ms": float(times_tensor.min().item()),
        "max_ms": float(times_tensor.max().item()),
        "median_ms": float(times_tensor.median().item()),
        "throughput_samples_per_sec": float(1000 / times_tensor.mean().item()),
    }


def profile_training_step(  # noqa: PLR0913
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    sample_input: torch.Tensor,
    sample_target: torch.Tensor,
    num_iterations: int = 50,
    warmup_iterations: int = 5,
) -> Dict[str, Any]:
    """
    Profile complete training step (forward + backward + optimizer).

    Args:
        model: PyTorch model to profile
        optimizer: Optimizer instance
        loss_fn: Loss function
        sample_input: Sample input tensor
        sample_target: Sample target tensor
        num_iterations: Number of profiling iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with timing statistics
    """
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)
    sample_target = sample_target.to(device)
    model.train()

    def step() -> float:
        optimizer.zero_grad()
        output = model(sample_input)
        loss = loss_fn(output, sample_target)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    # Warmup
    for _ in range(warmup_iterations):
        step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Profile
    times: List[float] = []
    losses: List[float] = []
    memory_used: List[float] = []

    for _ in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            loss_val = step()
            end_evt.record()
            torch.cuda.synchronize()
            times.append(float(start_evt.elapsed_time(end_evt)))
            memory_used.append(float(torch.cuda.max_memory_allocated() / (1024 * 1024)))
        else:
            start_time = time.perf_counter()
            loss_val = step()
            times.append(float((time.perf_counter() - start_time) * 1000))
        losses.append(loss_val)

    times_tensor = torch.tensor(times)
    return {
        "mean_ms": float(times_tensor.mean().item()),
        "std_ms": float(times_tensor.std().item()),
        "min_ms": float(times_tensor.min().item()),
        "max_ms": float(times_tensor.max().item()),
        "throughput_steps_per_sec": float(1000 / times_tensor.mean().item()),
        "avg_loss": sum(losses) / len(losses),
        "peak_memory_mb": max(memory_used) if memory_used else 0.0,
    }
