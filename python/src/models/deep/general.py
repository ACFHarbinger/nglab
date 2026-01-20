"""
General Neural Network Architectures Access Module.

Exports MLP, Perceptron, RBF, PINN, NeuralODE, and ELM architectures.
"""

from .general.elm import ELM
from .general.mlp import MLP
from .general.node import NeuralODE
from .general.perceptron import Perceptron
from .general.pinn import PINN
from .general.rbf import RBF

__all__ = [
    "ELM",
    "MLP",
    "PINN",
    "RBF",
    "NeuralODE",
    "Perceptron",
]
