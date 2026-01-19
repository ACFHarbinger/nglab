"""
GPU Profiling Module for NGLab.

Provides tools for profiling GPU utilization, memory usage, and training performance.
"""

from python.src.utils.profiling.cuda_profiler import (
    CUDAProfiler,
    ProfilerConfig,
    ProfilingResult,
    GPUMemoryStats,
    profile_model_forward,
    profile_training_step,
    get_gpu_memory_stats,
)
from python.src.utils.profiling.benchmark import (
    GPUBenchmark,
    BenchmarkResult,
    run_inference_benchmark,
    run_training_benchmark,
)
from python.src.utils.profiling.gpu_optimization import (
    MemoryPool,
    TransferProfile,
    TransferProfiler,
    GPUMemoryOptimizer,
    enable_memory_efficient_attention,
    optimize_for_inference,
    get_gpu_optimization_recommendations,
)

__all__ = [
    # CUDA Profiler
    "CUDAProfiler",
    "ProfilerConfig",
    "ProfilingResult",
    "GPUMemoryStats",
    "profile_model_forward",
    "profile_training_step",
    "get_gpu_memory_stats",
    # Benchmarks
    "GPUBenchmark",
    "BenchmarkResult",
    "run_inference_benchmark",
    "run_training_benchmark",
    # GPU Optimization
    "MemoryPool",
    "TransferProfile",
    "TransferProfiler",
    "GPUMemoryOptimizer",
    "enable_memory_efficient_attention",
    "optimize_for_inference",
    "get_gpu_optimization_recommendations",
]
