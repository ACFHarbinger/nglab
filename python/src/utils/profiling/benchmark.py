"""
GPU Benchmark Suite for NGLab.

Provides standardized benchmarks for measuring model performance
across different configurations and hardware.
"""

import gc
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    timestamp: str

    # Timing metrics (milliseconds)
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Throughput
    throughput_samples_per_sec: float
    throughput_batches_per_sec: float

    # Memory metrics (MB)
    peak_memory_mb: float
    avg_memory_mb: float

    # Configuration
    batch_size: int
    sequence_length: int
    num_iterations: int
    device: str
    dtype: str
    mixed_precision: bool = False

    # Hardware info
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    cuda_version: str = ""

    # Extra metrics
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, path: str | None = None) -> str:
        """Export to JSON."""
        data = self.to_dict()
        if path:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        return json.dumps(data, indent=2)


class GPUBenchmark:
    """
    GPU Benchmark suite for model performance testing.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        output_dir: str = "./benchmark_results",
    ) -> None:
        """Initialize GPU Benchmark."""
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[BenchmarkResult] = []

        # Collect hardware info
        self._hardware_info = self._get_hardware_info()

    def _get_hardware_info(self) -> dict[str, Any]:
        """Collect hardware information."""
        info = {
            "gpu_name": "",
            "gpu_memory_gb": 0.0,
            "cuda_version": "",
        }
        if torch.cuda.is_available() and "cuda" in self.device:
            device_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
            props = torch.cuda.get_device_properties(device_idx)
            info["gpu_name"] = props.name
            info["gpu_memory_gb"] = props.total_memory / (1024**3)
            info["cuda_version"] = torch.version.cuda or ""
        return info

    def run_inference(  # noqa: PLR0913
        self,
        input_shape: tuple[int, ...],
        batch_sizes: list[int] | None = None,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        dtype: torch.dtype = torch.float32,
        mixed_precision: bool = False,
    ) -> list[BenchmarkResult]:
        """
        Run inference benchmarks across different batch sizes.
        """
        if batch_sizes is None:
            batch_sizes = [1, 8, 32, 64]
        results = []
        self.model.eval()

        for batch_size in batch_sizes:
            result = self._benchmark_inference(
                input_shape=input_shape,
                batch_size=batch_size,
                num_iterations=num_iterations,
                warmup_iterations=warmup_iterations,
                dtype=dtype,
                mixed_precision=mixed_precision,
            )
            results.append(result)
            self.results.append(result)

            print(
                f"Batch {batch_size:>4d}: "
                f"Mean: {result.mean_latency_ms:>8.2f}ms, "
                f"P95: {result.p95_latency_ms:>8.2f}ms, "
                f"Throughput: {result.throughput_samples_per_sec:>8.1f} samples/sec"
            )

        return results

    def _benchmark_inference(  # noqa: PLR0913
        self,
        input_shape: tuple[int, ...],
        batch_size: int,
        num_iterations: int,
        warmup_iterations: int,
        dtype: torch.dtype,
        mixed_precision: bool,
    ) -> BenchmarkResult:
        """Run single inference benchmark."""
        sample_input = torch.randn(
            batch_size, *input_shape, dtype=dtype, device=self.device
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Warmup
        with torch.no_grad():
            if mixed_precision and torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    for _ in range(warmup_iterations):
                        _ = self.model(sample_input)
            else:
                for _ in range(warmup_iterations):
                    _ = self.model(sample_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        times: list[float] = []
        memory_readings: list[float] = []

        with torch.no_grad():
            for _ in range(num_iterations):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)

                    start_evt.record()
                    if mixed_precision:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            _ = self.model(sample_input)
                    else:
                        _ = self.model(sample_input)
                    end_evt.record()

                    torch.cuda.synchronize()
                    times.append(start_evt.elapsed_time(end_evt))
                    memory_readings.append(
                        torch.cuda.max_memory_allocated() / (1024 * 1024)
                    )
                else:
                    start_time = time.perf_counter()
                    if mixed_precision:
                        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                            _ = self.model(sample_input)
                    else:
                        _ = self.model(sample_input)
                    times.append((time.perf_counter() - start_time) * 1000)

        times_tensor = torch.tensor(times)
        sorted_times = times_tensor.sort().values

        return BenchmarkResult(
            name=f"inference_batch{batch_size}",
            timestamp=datetime.now().isoformat(),
            mean_latency_ms=times_tensor.mean().item(),
            std_latency_ms=times_tensor.std().item(),
            min_latency_ms=times_tensor.min().item(),
            max_latency_ms=times_tensor.max().item(),
            p50_latency_ms=sorted_times[int(0.50 * len(sorted_times))].item(),
            p95_latency_ms=sorted_times[int(0.95 * len(sorted_times))].item(),
            p99_latency_ms=sorted_times[int(0.99 * len(sorted_times))].item(),
            throughput_samples_per_sec=batch_size * 1000 / times_tensor.mean().item(),
            throughput_batches_per_sec=1000 / times_tensor.mean().item(),
            peak_memory_mb=max(memory_readings) if memory_readings else 0.0,
            avg_memory_mb=(
                sum(memory_readings) / len(memory_readings) if memory_readings else 0.0
            ),
            batch_size=batch_size,
            sequence_length=input_shape[0] if len(input_shape) > 0 else 0,
            num_iterations=num_iterations,
            device=self.device,
            dtype=str(dtype),
            mixed_precision=mixed_precision,
            **self._hardware_info,
        )

    def run_training(  # noqa: PLR0913
        self,
        input_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
        loss_fn: Callable[..., torch.Tensor],
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        batch_sizes: list[int] | None = None,
        num_iterations: int = 50,
        warmup_iterations: int = 5,
        dtype: torch.dtype = torch.float32,
        mixed_precision: bool = False,
    ) -> list[BenchmarkResult]:
        """
        Run training benchmarks across different batch sizes.
        """
        if batch_sizes is None:
            batch_sizes = [8, 32, 64]
        results = []

        for batch_size in batch_sizes:
            result = self._benchmark_training(
                input_shape=input_shape,
                target_shape=target_shape,
                loss_fn=loss_fn,
                optimizer_class=optimizer_class,
                batch_size=batch_size,
                num_iterations=num_iterations,
                warmup_iterations=warmup_iterations,
                dtype=dtype,
                mixed_precision=mixed_precision,
            )
            results.append(result)
            self.results.append(result)

            print(
                f"Batch {batch_size:>4d}: "
                f"Mean: {result.mean_latency_ms:>8.2f}ms, "
                f"P95: {result.p95_latency_ms:>8.2f}ms, "
                f"Throughput: {result.throughput_samples_per_sec:>8.1f} samples/sec, "
                f"Peak Mem: {result.peak_memory_mb:>8.1f}MB"
            )

        return results

    def _benchmark_training(  # noqa: PLR0913 PLR0915
        self,
        input_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
        loss_fn: Callable[..., torch.Tensor],
        optimizer_class: type[torch.optim.Optimizer],
        batch_size: int,
        num_iterations: int,
        warmup_iterations: int,
        dtype: torch.dtype,
        mixed_precision: bool,
    ) -> BenchmarkResult:
        """Run single training benchmark."""
        self.model.train()

        optimizer = cast(Any, optimizer_class)(self.model.parameters(), lr=1e-4)
        scaler = (
            torch.cuda.amp.GradScaler()
            if mixed_precision and torch.cuda.is_available()
            else None
        )

        sample_input = torch.randn(
            batch_size, *input_shape, dtype=dtype, device=self.device
        )
        sample_target = torch.randn(
            batch_size, *target_shape, dtype=dtype, device=self.device
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        def train_step(
            opt: torch.optim.Optimizer, sclr: torch.cuda.amp.GradScaler | None
        ) -> None:
            """Execute a single training step for benchmarking."""
            opt.zero_grad()
            if mixed_precision and sclr is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(sample_input)
                    loss = loss_fn(output, sample_target)
                sclr.scale(loss).backward()
                sclr.step(opt)
                sclr.update()
            else:
                output = self.model(sample_input)
                loss = loss_fn(output, sample_target)
                loss.backward()
                opt.step()

        # Warmup
        for _ in range(warmup_iterations):
            train_step(optimizer, scaler)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        times: list[float] = []
        memory_readings: list[float] = []

        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)

                start_evt.record()
                train_step(optimizer, scaler)
                end_evt.record()

                torch.cuda.synchronize()
                times.append(start_evt.elapsed_time(end_evt))
                memory_readings.append(
                    torch.cuda.max_memory_allocated() / (1024 * 1024)
                )
            else:
                start_time = time.perf_counter()
                train_step(optimizer, scaler)
                times.append((time.perf_counter() - start_time) * 1000)

        times_tensor = torch.tensor(times)
        sorted_times = times_tensor.sort().values

        # Cleanup
        del optimizer
        if scaler:
            del scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return BenchmarkResult(
            name=f"training_batch{batch_size}{'_amp' if mixed_precision else ''}",
            timestamp=datetime.now().isoformat(),
            mean_latency_ms=times_tensor.mean().item(),
            std_latency_ms=times_tensor.std().item(),
            min_latency_ms=times_tensor.min().item(),
            max_latency_ms=times_tensor.max().item(),
            p50_latency_ms=sorted_times[int(0.50 * len(sorted_times))].item(),
            p95_latency_ms=sorted_times[int(0.95 * len(sorted_times))].item(),
            p99_latency_ms=sorted_times[int(0.99 * len(sorted_times))].item(),
            throughput_samples_per_sec=batch_size * 1000 / times_tensor.mean().item(),
            throughput_batches_per_sec=1000 / times_tensor.mean().item(),
            peak_memory_mb=max(memory_readings) if memory_readings else 0.0,
            avg_memory_mb=(
                sum(memory_readings) / len(memory_readings) if memory_readings else 0.0
            ),
            batch_size=batch_size,
            sequence_length=input_shape[0] if len(input_shape) > 0 else 0,
            num_iterations=num_iterations,
            device=self.device,
            dtype=str(dtype),
            mixed_precision=mixed_precision,
            **self._hardware_info,
        )

    def save_results(self, filename: str | None = None) -> str:
        """Save all benchmark results to JSON file."""
        if filename is None:
            filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        path = self.output_dir / filename
        data = {
            "timestamp": datetime.now().isoformat(),
            "hardware": self._hardware_info,
            "results": [r.to_dict() for r in self.results],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return str(path)

    def compare_with_baseline(self, baseline_path: str) -> dict[str, Any]:
        """
        Compare current results with a baseline.
        """
        with open(baseline_path) as f:
            baseline = json.load(f)

        baseline_results = {r["name"]: r for r in baseline["results"]}
        comparisons: dict[str, Any] = {}

        for result in self.results:
            if result.name in baseline_results:
                base = baseline_results[result.name]
                change = (
                    (result.mean_latency_ms - base["mean_latency_ms"])
                    / base["mean_latency_ms"]
                    * 100
                )
                comparisons[result.name] = {
                    "latency_change_percent": change,
                    "throughput_change_percent": -change,
                    "baseline_latency_ms": base["mean_latency_ms"],
                    "current_latency_ms": result.mean_latency_ms,
                }

        return comparisons


def run_inference_benchmark(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str = "cuda",
    batch_sizes: list[int] | None = None,
    **kwargs: Any,
) -> list[BenchmarkResult]:
    """Convenience function for running inference benchmarks."""
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64]
    benchmark = GPUBenchmark(model, device=device)
    return benchmark.run_inference(input_shape, batch_sizes=batch_sizes, **kwargs)


def run_training_benchmark(  # noqa: PLR0913
    model: nn.Module,
    input_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    loss_fn: Callable[..., torch.Tensor],
    device: str = "cuda",
    batch_sizes: list[int] | None = None,
    **kwargs: Any,
) -> list[BenchmarkResult]:
    """Convenience function for running training benchmarks."""
    if batch_sizes is None:
        batch_sizes = [8, 32, 64]
    benchmark = GPUBenchmark(model, device=device)
    return benchmark.run_training(
        input_shape, target_shape, loss_fn, batch_sizes=batch_sizes, **kwargs
    )
