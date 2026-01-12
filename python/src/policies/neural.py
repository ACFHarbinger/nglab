import torch
from tensordict import TensorDict
from .base import Policy

class NeuralPolicy(Policy):
    """
    Policy that wraps a PyTorch/TorchRL module.
    """
    def __init__(self, model, cfg=None):
        super().__init__(cfg)
        self.model = model # Expecting a TensorDictModule or similar
        self.device = self.cfg.get('device', 'cpu')

    def act(self, observation):
        # Expecting observation to be compatible with model input
        # If model expects TensorDict:
        if isinstance(observation, dict) and not isinstance(observation, TensorDict):
             observation = TensorDict(observation, batch_size=[])
        
        with torch.no_grad():
            output = self.model(observation)
            
        # Extract action
        # Assuming model outputs 'action' key or we need to sample distribution
        if 'action' in output.keys():
            return output['action']
        elif 'logits' in output.keys():
            return output['logits'].argmax(dim=-1)
        
        return output
