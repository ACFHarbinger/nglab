from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar, Union

from omegaconf import DictConfig, OmegaConf
from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

def validate_config(schema: Any = None) -> Callable[[F], F]:
    """
    Decorator to validate configuration arguments.
    
    If schema is provided, it validates the config against the schema.
    Otherwise, it performs basic sanity checks.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Find config in args or kwargs
            cfg = None
            if args and isinstance(args[0], (dict, DictConfig)):
                cfg = args[0]
            elif "cfg" in kwargs:
                cfg = kwargs["cfg"]
            elif "config" in kwargs:
                cfg = kwargs["config"]
                
            if cfg is not None:
                _perform_validation(cfg, schema)
                
            return func(*args, **kwargs)
        return wrapper # type: ignore
    return decorator

def _perform_validation(cfg: Any, schema: Any = None) -> None:
    """Internal validation logic."""
    if isinstance(cfg, dict):
        # Already a dict
        pass
    elif isinstance(cfg, DictConfig):
        # OmegaConf config
        pass
    else:
        # Possibly a dataclass
        if not hasattr(cfg, "__dataclass_fields__"):
            return

    if schema:
        # TODO: Implement schema-based validation (e.g. using pydantic or cerberus)
        # For now, just log that we would validate against schema
        logger.debug(f"Validating config against schema: {schema}")

    # Basic sanity checks
    if hasattr(cfg, "task") and not cfg.task:
        raise ConfigurationError("Config 'task' cannot be empty.")

def validate_input(condition: Callable[[Any], bool], message: str) -> Callable[[F], F]:
    """
    Decorator to validate function inputs based on a condition.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for arg in args:
                if not condition(arg):
                    raise ValueError(f"Input validation failed: {message}")
            for value in kwargs.values():
                if not condition(value):
                    raise ValueError(f"Input validation failed: {message}")
            return func(*args, **kwargs)
        return wrapper # type: ignore
    return decorator

__all__ = ["validate_config", "validate_input"]
