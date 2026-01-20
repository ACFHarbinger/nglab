import os
from pathlib import Path

# Paths
path = Path(os.getcwd())
parts = path.parts
ROOT_DIR = Path(*parts[: parts.index("nglab") + 1])
ICON_FILE = os.path.join(ROOT_DIR, "assets", "images", "logo-nglab.png")

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


# Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", "0.01"))
REDIS_HOST = os.getenv("NGLAB_REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("NGLAB_REDIS_PORT", "6379")
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"
CACHE_TTL = 60  # seconds
