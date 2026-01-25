"""
Autoencoder Architectures.

A collection of Autoencoder implementations for representation learning:
- Vanilla AutoEncoder (AE)
- Denoising AutoEncoder (DAE)
- Sparse AutoEncoder (SAE)
- Stacked AutoEncoder
- Variational AutoEncoder (VAE)
"""

from .ae import AutoEncoder
from .dae import DenoisingAE
from .sae import SparseAE
from .stacked_ae import StackedAutoEncoder
from .vae import VAE

__all__ = [
    "VAE",
    "AutoEncoder",
    "DenoisingAE",
    "SparseAE",
    "StackedAutoEncoder",
]
