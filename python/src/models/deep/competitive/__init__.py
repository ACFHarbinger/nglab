"""
Competitive Learning Networks.

Implementations of competitive learning algorithms including:
- Self-Organizing Maps (SOM) / Kohonen Maps
- Learning Vector Quantization (LVQ)
"""

from .lvq import LVQ
from .som import KohonenMap

__all__ = [
    "LVQ",
    "KohonenMap",
]
