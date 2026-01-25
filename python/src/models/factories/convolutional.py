
from __future__ import annotations

from typing import Any

import torch.nn as nn

from python.src.models.convolutional.cnn import CNN
from python.src.models.convolutional.resnet import ResNet
from python.src.models.convolutional.capsule import CapsuleNet
from python.src.models.convolutional.dcign import DCIGN
from python.src.models.convolutional.dcn import DCN
from python.src.models.convolutional.deconv import DeconvNet
from python.src.models.factories.base import NeuralComponentFactory


class ConvolutionalFactory(NeuralComponentFactory):
    """Factory for convolutional networks."""

    @staticmethod
    def get_component(name: str, **kwargs: Any) -> nn.Module:
        """Get convolutional model by name."""
        name = name.lower()
        if "resnet" in name:
            return ResNet(**kwargs)
        elif "capsule" in name:
            return CapsuleNet(**kwargs)
        elif "dcign" in name:
            return DCIGN(**kwargs)
        elif "deconv" in name:
            return DeconvNet(**kwargs)
        elif "dcn" in name:
            return DCN(**kwargs)
        elif "cnn" in name:
            return CNN(**kwargs)
        else:
            raise ValueError(
                f"Unknown convolutional model: {name}. "
                f"Available: cnn, resnet, capsule, dcign, deconv, dcn"
            )
