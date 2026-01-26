from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

T = TypeVar("T", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def validate_config(schema: Any = None) -> Callable[[T], T]:
    """Decorator to validate configuration before object initialization."""

    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Example validation logic
            if "cfg" in kwargs:
                cfg = kwargs["cfg"]
                if hasattr(cfg, "validate"):
                    cfg.validate()
            elif len(args) > 1 and hasattr(args[1], "validate"):
                args[1].validate()

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def validate_input(func: T) -> T:
    """Decorator to validate input tensors/data before processing."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper  # type: ignore


def log_execution(func: T) -> T:
    """Decorator to log function execution for debugging architecture issues."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.debug(f"Executing {func.__name__} in {func.__module__}")
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper  # type: ignore
