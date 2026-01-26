
from __future__ import annotations

import yaml
from dataclasses import asdict, dataclass
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T", bound="BaseConfig")

__all__ = ["BaseConfig", "deep_sanitize"]


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
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create configuration from a dictionary, recursively handling nested configs."""
        kwargs = {}
        for k, v in data.items():
            if k in cls.__dataclass_fields__:
                field_type = cls.__dataclass_fields__[k].type
                # Handle nested BaseConfig
                if (
                    isinstance(field_type, type)
                    and issubclass(field_type, BaseConfig)
                    and isinstance(v, dict)
                ):
                    kwargs[k] = field_type.from_dict(v)
                else:
                    kwargs[k] = v
        return cls(**kwargs)

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
