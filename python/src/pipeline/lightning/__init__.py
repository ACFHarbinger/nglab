"""
PyTorch Lightning Modules for NGLab.
"""
from .base import BaseModule
from .rl_module import RLLightningModule
from .sl_module import SLLightningModule
from .gan_module import GANLightningModule
from .self_supervised import SelfSupervisedModule
from .semi_supervised import SemiSupervisedModule
from .unsupervised import UnsupervisedModule

__all__ = [
    'BaseModule',
    'RLLightningModule',
    'SLLightningModule',
    'GANLightningModule',
    'SelfSupervisedModule',
    'SemiSupervisedModule',
    'UnsupervisedModule'
]
