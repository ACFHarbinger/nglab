from __future__ import annotations

from typing import Any, Dict, List, Union
from omegaconf import DictConfig, ListConfig

def deep_sanitize(cfg: Union[DictConfig, ListConfig, Dict[str, Any], List[Any], Any]) -> Any:
    """
    Convert DictConfig/ListConfig to primitive Python types.

    CRITICAL: Always use this before passing config to PyTorch Lightning
    modules to avoid YAML serialization errors.

    Args:
        cfg: Configuration object (DictConfig, ListConfig, dict, or list).

    Returns:
        The configuration converted to primitive Python types (dict, list, etc.).

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({"lr": 0.001, "layers": [64, 128]})
        >>> sanitized = deep_sanitize(cfg)
        >>> type(sanitized["layers"])
        <class 'list'>
    """
    if isinstance(cfg, DictConfig):
        return {str(k): deep_sanitize(v) for k, v in cfg.items()}
    elif isinstance(cfg, ListConfig):
        return [deep_sanitize(v) for v in cfg]
    elif isinstance(cfg, dict):
        return {str(k): deep_sanitize(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [deep_sanitize(v) for v in cfg]
    else:
        return cfg

def sanitize_and_inject(cfg: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Sanitize a configuration and inject additional non-serializable objects.
    
    Args:
        cfg: The configuration to sanitize.
        **kwargs: Additional objects to inject after sanitization.
        
    Returns:
        A dictionary containing the sanitized configuration and injected objects.
    """
    sanitized = deep_sanitize(cfg)
    if not isinstance(sanitized, dict):
        if not sanitized:
            sanitized = {}
        else:
            # If it's not a dict, we can't inject into it easily
            # But usually top-level configs passed to modules are dicts
            return sanitized
    
    sanitized.update(kwargs)
    return sanitized
