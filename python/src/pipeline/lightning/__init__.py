"""
PyTorch Lightning Modules for NGLab.
"""
from .base import BaseModule
from .self_supervised import SelfSupervisedModule
from .semi_supervised import SemiSupervisedModule
from .unsupervised import UnsupervisedModule

__all__ = [
    'BaseModule',
    'SelfSupervisedModule',
    'SemiSupervisedModule',
    'UnsupervisedModule'
]
