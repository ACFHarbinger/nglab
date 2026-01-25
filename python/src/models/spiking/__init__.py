"""
Spiking Neural Network (SNN) Components.

Provides implementations of Spiking Neural Networks elements such as Leaky Integrate-and-Fire (LIF) cells
and surrogate gradient functions for training SNNs with backpropagation.
"""

from .snn import SNN, LIFCell, SurrogateHeaviside, surrogate_heaviside

__all__ = [
    "SNN",
    "LIFCell",
    "SurrogateHeaviside",
    "surrogate_heaviside",
]
