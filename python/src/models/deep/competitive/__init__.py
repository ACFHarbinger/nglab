"""
Competitive Learning Networks.

Implementations of competitive learning algorithms including:
- Self-Organizing Maps (SOM) / Kohonen Maps
- Learning Vector Quantization (LVQ)
"""
from .som import KohonenMap
from .lvq import LVQ

__all__ = [
    "KohonenMap",
    "LVQ",
]
