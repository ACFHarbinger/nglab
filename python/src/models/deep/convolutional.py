"""
Convolutional Neural Network (CNN) Variants Access Module.

Exports CNNs, DeepConvNet, DeconvNets, ResNets, and Capsule Networks.
"""
from .convolutional.cnn import RollingWindowCNN
from .convolutional.dcn import DeepConvNet
from .convolutional.deconv import DeconvNet, AutoDeconvNet
from .convolutional.dcign import DCIGN
from .convolutional.resnet import DeepResNet
from .convolutional.capsule import CapsuleLayer

__all__ = [
    "RollingWindowCNN",
    "DeepConvNet",
    "DeconvNet",
    "AutoDeconvNet",
    "DCIGN",
    "DeepResNet",
    "CapsuleLayer",
]
