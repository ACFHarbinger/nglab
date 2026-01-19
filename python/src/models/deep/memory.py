"""
Memory Augmented Neural Networks (MANNs) Access Module.

Exports memory-augmented architectures like DNC and NTM.
"""

from .memory.dnc import DNC
from .memory.ntm import NTM

__all__ = [
    "DNC",
    "NTM",
]
