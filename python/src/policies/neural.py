"""
Neural Network Policy for NGLab.

Wraps PyTorch or TorchRL modules to provide a consistent 'act' interface
for model-based trading strategies.
"""

from typing import Any

import torch
from tensordict import TensorDict
from torch import nn

from .base import Policy


class NeuralPolicy(Policy):
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
        super().__init__(cfg)
        self.model = model  # Expecting a TensorDictModule or similar
        self.device = self.cfg.get("device", "cpu")

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
