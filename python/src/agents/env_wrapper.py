"""
TorchRL Environment Wrapper for NGLab.

Adapts the standard Gymnasium TradingEnv for use within the TorchRL framework,
handling TensorDict mapping and spec inference.
"""
from torchrl.envs import GymWrapper
import gymnasium as gym
from env.trading_env import TradingEnv

class TradingEnvWrapper(GymWrapper):
    """
    Wrapper for TradingEnv to be compatible with TorchRL.
    Ensures observations and actions are mapped to TensorDicts.
    """
    def __init__(self, env=None, device="cpu", **kwargs):
        """
        Initialize the environment wrapper.

        Args:
            env (gym.Env, optional): An existing Gymnasium environment.
            device (str): Device to use for tensors.
            **kwargs: Arguments to pass to TradingEnv if 'env' is None.
        """
        if env is None:
            env = TradingEnv(**kwargs)
        
        super().__init__(env, device=device)
        
    def _make_specs(self, env: gym.Env) -> None:
        """
        Infer and validate environment specs.
        """
        super()._make_specs(env)
        # Custom spec adjustments if necessary
        # For example ensuring observation_spec is correctly named 'observation'
        
        # Verify specs are present
        if self.observation_spec is None:
             raise ValueError("Observation spec could not be inferred.")
        if self.action_spec is None:
             raise ValueError("Action spec could not be inferred.")
