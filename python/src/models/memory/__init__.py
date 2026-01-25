"""
Memory-Augmented Neural Networks (MANNs).

Architectures with external memory access:
- Neural Turing Machines (NTM)
- Differentiable Neural Computers (DNC)
"""

from .dnc import DNC
from .ntm import NTM

__all__ = [
    "DNC",
    "NTM",
]
