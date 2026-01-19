"""
General Neural Network Architectures Access Module.

Exports MLP, Perceptron, RBF, PINN, NeuralODE, and ELM architectures.
"""
from .general.mlp import MLP
from .general.perceptron import Perceptron
from .general.rbf import RBF
from .general.pinn import PINN
from .general.node import NeuralODE
from .general.elm import ELM

__all__ = [
    "MLP",
    "Perceptron",
    "RBF",
    "PINN",
    "NeuralODE",
    "ELM",
]
