"""File and directory utility functions."""

import json
import os
from collections.abc import Callable
from typing import Any


def compose_dirpath(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to ensure the directory of the file path passed to the function exists.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper that ensures directory existence before calling the function."""
        path = args[0] if args else kwargs.get("json_path") or kwargs.get("dir_path")
        if path:
            os.makedirs(
                os.path.dirname(path) if os.path.isfile(path) else path, exist_ok=True
            )
        return func(*args, **kwargs)

    return wrapper


def read_json(path: str, lock: Any | None = None) -> Any:
    """
    Reads a JSON file, optionally with a lock.
    """
    if lock is not None:
        lock.acquire()
    try:
        with open(path) as f:
            return json.load(f)
    finally:
        if lock is not None:
            lock.release()
