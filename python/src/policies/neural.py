"""
Neural Network Policy for NGLab.

Wraps PyTorch or TorchRL modules to provide a consistent 'act' interface
for model-based trading strategies.
"""

from typing import Any

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torch import nn

from python.src.utils.registry import register_policy
from .base import Policy


@register_policy("neural")
class NeuralPolicy(TensorDictModuleBase, Policy):
    """
    Policy that wraps a PyTorch/TorchRL module.
    """

    def __init__(self, model: nn.Module, cfg: dict[str, Any] | None = None) -> None:
        """
        Initialize the neural policy.

        Args:
            model (nn.Module): The underlying neural network model.
            cfg (Dict, optional): Configuration dictionary.
        """
        TensorDictModuleBase.__init__(self)
        Policy.__init__(self, cfg)
        self.model = model  # Expecting a TensorDictModule or similar
        self.device = self.cfg.get("device", "cpu")

    def forward(self, x: Any) -> Any:
        """
        Forward pass for nn.Module compatibility.
        """
        return self.model(x)

    @property
    def in_keys(self) -> list[str]:
        return ["observation"]

    @property
    def out_keys(self) -> list[str]:
        return [
            "logits"
        ]  # Assuming the backbone outputs logits for the action distribution

    def act(self, observation: torch.Tensor | TensorDict | dict[str, Any]) -> Any:
        """
        Execute the model forward pass and extract an action.

        Args:
            observation (Any): Model input (tensor or Dict).

        Returns:
            Any: The predicted action.
        """
        # Expecting observation to be compatible with model input
        # If model expects TensorDict:
        if isinstance(observation, dict) and not isinstance(observation, TensorDict):
            observation = TensorDict(observation, batch_size=[])

        with torch.no_grad():
            output = self.model(observation)

        # Extract action
        # Assuming model outputs 'action' key or we need to sample distribution
        if hasattr(output, "keys"):
            if "action" in output.keys():
                return output["action"]
            elif "logits" in output.keys():
                return output["logits"].argmax(dim=-1)

        return output
