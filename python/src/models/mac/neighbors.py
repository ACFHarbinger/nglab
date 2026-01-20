"""Neighbors models facade."""

from .neighbors.knn import kNNModel
from .neighbors.lwl import LWLModel

__all__ = [
    "LWLModel",
    "kNNModel",
]
