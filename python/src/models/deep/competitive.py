"""
Competitive Learning Architectures Access Module.

Exports Self-Organizing Maps (KohonenMap) and Learning Vector Quantization (LVQ).
"""

from .competitive.lvq import LVQ
from .competitive.som import KohonenMap

__all__ = [
    "LVQ",
    "KohonenMap",
]
