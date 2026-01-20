"""
GPU Profiling Module for NGLab.

Provides tools for profiling GPU utilization, memory usage, and training performance.
"""

from python.src.utils.profiling.benchmark import (
    BenchmarkResult,
    GPUBenchmark,
    run_inference_benchmark,
    run_training_benchmark,
)
from python.src.utils.profiling.cuda_profiler import (
    CUDAProfiler,
    GPUMemoryStats,
    ProfilerConfig,
    ProfilingResult,
    get_gpu_memory_stats,
    profile_model_forward,
    profile_training_step,
)
from python.src.utils.profiling.gpu_optimization import (
    GPUMemoryOptimizer,
    MemoryPool,
    TransferProfile,
    TransferProfiler,
    enable_memory_efficient_attention,
    get_gpu_optimization_recommendations,
    optimize_for_inference,
)

__all__ = [
    "BenchmarkResult",
    # CUDA Profiler
    "CUDAProfiler",
    # Benchmarks
    "GPUBenchmark",
    "GPUMemoryOptimizer",
    "GPUMemoryStats",
    # GPU Optimization
    "MemoryPool",
    "ProfilerConfig",
    "ProfilingResult",
    "TransferProfile",
    "TransferProfiler",
    "enable_memory_efficient_attention",
    "get_gpu_memory_stats",
    "get_gpu_optimization_recommendations",
    "optimize_for_inference",
    "profile_model_forward",
    "profile_training_step",
    "run_inference_benchmark",
    "run_training_benchmark",
]
