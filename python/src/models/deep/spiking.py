"""
Spiking Neural Network (SNN) Components Access Module.

Exports SNN architecture and cells (LIFCell, SurrogateHeaviside).
"""
from .spiking.snn import SNN, LIFCell, SurrogateHeaviside, surrogate_heaviside

__all__ = [
    "SNN",
    "LIFCell",
    "SurrogateHeaviside",
    "surrogate_heaviside",
]
