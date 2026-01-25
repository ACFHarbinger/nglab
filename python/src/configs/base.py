
from __future__ import annotations

import yaml
from dataclasses import asdict, dataclass
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base configuration class with utility methods."""

    @classmethod
    def from_yaml(cls: Type[T], path: str) -> T:
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create configuration from a dictionary."""
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return asdict(self)


def deep_sanitize(cfg: Any) -> Any:
    """Recursively convert DictConfig/ListConfig to primitives."""
    from omegaconf import DictConfig, ListConfig
    if isinstance(cfg, DictConfig):
        return {k: deep_sanitize(v) for k, v in cfg.items()}
    if isinstance(cfg, ListConfig):
        return [deep_sanitize(v) for v in cfg]
    if isinstance(cfg, dict):
        return {k: deep_sanitize(v) for k, v in cfg.items()}
    if isinstance(cfg, list | tuple):
        return [deep_sanitize(v) for v in cfg]
    return cfg
