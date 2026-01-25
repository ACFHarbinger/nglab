
"""
PyTorch Lightning Modules for NGLab.
"""

from .base import BaseModule
from .diffusion_module import DiffusionLightningModule
from .gan_module import GANLightningModule
from .reinforcement_learning import RLLightningModule
from .self_supervised import SelfSupervisedModule
from .semi_supervised import SemiSupervisedModule
from .supervised_learning import SLLightningModule
from .unsupervised_learning import UnsupervisedModule
from .vae_module import VAELightningModule

__all__ = [
    "BaseModule",
    "DiffusionLightningModule",
    "GANLightningModule",
    "RLLightningModule",
    "SLLightningModule",
    "SelfSupervisedModule",
    "SemiSupervisedModule",
    "UnsupervisedModule",
    "VAELightningModule",
]
