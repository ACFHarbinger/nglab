from __future__ import annotations

import functools
import time
import logging
import psutil
import os
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

def profile_performance(func: F) -> F:
    """
    Decorator to profile execution time and memory usage of a function.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.perf_counter()
            end_mem = process.memory_info().rss / (1024 * 1024)
            
            duration = end_time - start_time
            mem_diff = end_mem - start_mem
            
            logger.info(
                f"Performance: {func.__name__} took {duration:.4f}s, "
                f"Memory: {start_mem:.2f}MB -> {end_mem:.2f}MB ({mem_diff:+.2f}MB)"
            )
            
        return result
    return wrapper # type: ignore

__all__ = ["profile_performance"]
