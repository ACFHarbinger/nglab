"""Connection factory for various neural network architectures."""
import torch.nn as nn

from .skip_connection import SkipConnection
from .hyper_connection import StaticHyperConnection, DynamicHyperConnection

class Connections(nn.Module):
    """
    Factory for creating connection modules.
    """
    def __init__(self):
        """Initializes the connections factory."""
        super(Connections, self).__init__()

def get_connection_module(module, embed_dim, connection_type='skip', **kwargs):
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
    if connection_type == 'skip':
        return SkipConnection(module)
    elif connection_type == 'static_hyper':
        return StaticHyperConnection(module, embed_dim, **kwargs)
    elif connection_type == 'dynamic_hyper':
        return DynamicHyperConnection(module, embed_dim, **kwargs)
    else:
        raise ValueError(f"Unknown connection type: {connection_type}")


