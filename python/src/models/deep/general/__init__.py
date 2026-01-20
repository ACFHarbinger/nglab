"""
General Neural Architectures.

Standard and specialized neural network components:
- Multilayer Perceptron (MLP)
- Simple Perceptron
- Radial Basis Function Networks (RBF)
- Physics-Informed Neural Networks (PINN)
- Neural ODEs (Ordinary Differential Equations)
- Extreme Learning Machines (ELM)
"""

from .mlp import MLP
from .perceptron import Perceptron
from .rbf import RBF
from .pinn import PINN
from .node import NeuralODE, odesolve
from .elm import ELM

__all__ = [
    "MLP",
    "Perceptron",
    "RBF",
    "PINN",
    "NeuralODE",
    "ELM",
]
