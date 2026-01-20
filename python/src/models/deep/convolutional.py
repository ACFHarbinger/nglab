"""
Convolutional Neural Network (CNN) Variants Access Module.

Exports CNNs, DeepConvNet, DeconvNets, ResNets, and Capsule Networks.
"""

from .convolutional.capsule import CapsuleLayer
from .convolutional.cnn import RollingWindowCNN
from .convolutional.dcign import DCIGN
from .convolutional.dcn import DeepConvNet
from .convolutional.deconv import AutoDeconvNet, DeconvNet
from .convolutional.resnet import DeepResNet

__all__ = [
    "DCIGN",
    "AutoDeconvNet",
    "CapsuleLayer",
    "DeconvNet",
    "DeepConvNet",
    "DeepResNet",
    "RollingWindowCNN",
]
