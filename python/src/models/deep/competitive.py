"""
Competitive Learning Architectures Access Module.

Exports Self-Organizing Maps (KohonenMap) and Learning Vector Quantization (LVQ).
"""

from .competitive.som import KohonenMap
from .competitive.lvq import LVQ

__all__ = [
    "KohonenMap",
    "LVQ",
]
