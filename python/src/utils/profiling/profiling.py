"""
Performance profiling utilities for NGLab.

Provides decorators for:
- @profile: cProfile-based function profiling
- @timeit: High-resolution execution time measurement
- @memory_profile: Memory usage tracking

All results are logged and can be saved to disk for analysis.
"""

from __future__ import annotations

import cProfile
import io
import logging
import os
import pstats
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast

import psutil

__all__ = ["profile", "timeit", "memory_profile"]

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def profile(output_dir: str = "./profiles") -> Callable[[F], F]:
    """
    Decorator to profile function execution using cProfile.
    Writes results to a .prof file in the specified directory.
    """

    def decorator(func: F) -> F:
        """Inner decorator that wraps the function."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper that profiles function execution."""
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            prof = cProfile.Profile()
            prof.enable()
            result = func(*args, **kwargs)
            prof.disable()

            # Save raw stats
            stats_file = out_path / f"{func.__name__}.prof"
            prof.dump_stats(str(stats_file))

            # Log summary
            s = io.StringIO()
            ps = pstats.Stats(prof, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats(20)
            logger.info(f"Performance profile for {func.__name__}:\n{s.getvalue()}")

            return result

        return cast(F, wrapper)

    return decorator


def timeit(func: F) -> F:
    """
    Decorator to measure function execution time with high resolution.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper that measures execution time."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"Function {func.__name__} took {duration:.6f} seconds")
        return result

    return cast(F, wrapper)


def memory_profile(func: F) -> F:
    """
    Decorator to profile memory usage of a function.
    Logs RSS memory before and after execution.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper that measures memory usage."""
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        result = func(*args, **kwargs)
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        diff = mem_after - mem_before
        logger.info(
            f"Memory profile for {func.__name__}: "
            f"Before: {mem_before:.2f}MB, After: {mem_after:.2f}MB, "
            f"Diff: {diff:+.2f}MB"
        )
        return result

    return cast(F, wrapper)
