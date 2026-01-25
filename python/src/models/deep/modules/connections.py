"""Connection factory for various neural network architectures."""

from typing import Any

from torch import nn

from .hyper_connection import DynamicHyperConnection, StaticHyperConnection
from .skip_connection import SkipConnection


class Connections(nn.Module):
    """
    Factory for creating connection modules.
    """

    def __init__(self) -> None:
        """Initializes the connections factory."""
        super().__init__()


def get_connection_module(
    module: nn.Module, embed_dim: int, connection_type: str = "skip", **kwargs: Any
) -> nn.Module:
    """
    Returns a connection module for the given type.

    Args:
        module: The sub-module to wrap.
        embed_dim: Embedding dimension.
        connection_type: Type of connection ('skip', 'static_hyper', 'dynamic_hyper').
        **kwargs: Additional arguments for the connection module.

    Returns:
        The connection module.
    """
    if connection_type == "skip":
        return SkipConnection(module)
    elif connection_type == "static_hyper":
        return StaticHyperConnection(module, embed_dim, **kwargs)
    elif connection_type == "dynamic_hyper":
        return DynamicHyperConnection(module, embed_dim, **kwargs)
    else:
        raise ValueError(f"Unknown connection type: {connection_type}")
