"""
Global constants and configuration definitions for NGLab.

This module now re-exports constants from python.src.constants to maintain
backward compatibility.
"""

from python.src.constants.paths import ICON_FILE, ROOT_DIR
from python.src.constants.system import (
    CACHE_TTL,
    CORE_LOCK_WAIT_TIME,
    LOCK_TIMEOUT,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_URL,
    update_lock_wait_time,
)
from python.src.constants.training import DEFAULT_BATCH_SIZE, BATCH_TIMEOUT

# Re-export path parts if needed, though they were derived from cwd which is flaky.
# We will skip 'path' and 'parts' unless strictly necessary (inference.py doesn't seem to use them).
