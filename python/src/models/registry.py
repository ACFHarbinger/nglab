from __future__ import annotations

from ..utils.registry import MODEL_REGISTRY, register_model

def get_model(name: str) -> type:
    """Get model class by name."""
    return MODEL_REGISTRY.get(name)

__all__ = ["MODEL_REGISTRY", "register_model", "get_model"]
