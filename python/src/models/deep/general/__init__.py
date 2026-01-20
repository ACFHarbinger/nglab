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

from .elm import ELM
from .mlp import MLP
from .node import NeuralODE, odesolve  # noqa: F401
from .perceptron import Perceptron
from .pinn import PINN
from .rbf import RBF

__all__ = [
    "ELM",
    "MLP",
    "PINN",
    "RBF",
    "NeuralODE",
    "Perceptron",
]
