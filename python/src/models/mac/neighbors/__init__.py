"""Neighbors models package."""

from .knn import kNNModel
from .lwl import LWLModel

__all__ = [
    "LWLModel",
    "kNNModel",
]
