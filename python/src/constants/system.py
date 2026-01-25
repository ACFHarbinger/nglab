
from __future__ import annotations

import os

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hardware
DEFAULT_DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
SEED = 42

# API Keys (Placeholders)
POLYMARKET_API_KEY = os.environ.get("POLYMARKET_API_KEY", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

# Multi-core processing settings
CORE_LOCK_WAIT_TIME = 10
LOCK_TIMEOUT = CORE_LOCK_WAIT_TIME


def update_lock_wait_time(num_cpu_cores: int | None = None) -> int:
    """
    Updates the global LOCK_TIMEOUT based on the number of CPU cores.

    Returns:
        The new (or default) value of LOCK_TIMEOUT.
    """
    global LOCK_TIMEOUT  # noqa: PLW0603
    if num_cpu_cores is None:
        LOCK_TIMEOUT = CORE_LOCK_WAIT_TIME
    else:
        LOCK_TIMEOUT = CORE_LOCK_WAIT_TIME * num_cpu_cores
    return LOCK_TIMEOUT


# Infrastructure / Redis
REDIS_HOST = os.getenv("NGLAB_REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("NGLAB_REDIS_PORT", "6379")
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"
CACHE_TTL = 60  # seconds
