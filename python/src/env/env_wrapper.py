"""
TorchRL Environment Wrapper for NGLab.

Adapts the standard Gymnasium TradingEnv for use within the TorchRL framework,
handling TensorDict mapping and spec inference.
"""

import gymnasium as gym
from .trading_env import TradingEnv
from .vectorized_env import make_vec_env
from torchrl.envs import GymWrapper


class TradingEnvWrapper(GymWrapper):
    """
    Wrapper for TradingEnv to be compatible with TorchRL.
    Ensures observations and actions are mapped to TensorDicts.
    """

    def __init__(self, env=None, device="cpu", num_envs=1, **kwargs):
        """
        Initialize the environment wrapper.

        Args:
            env (gym.Env, optional): An existing Gymnasium environment.
            device (str): Device to use for tensors.
            num_envs (int): Number of parallel environments to create if env is None.
            **kwargs: Arguments to pass to TradingEnv/make_vec_env if 'env' is None.
        """
        batch_size = None
        if env is None:
            if num_envs > 1:
                env = make_vec_env(num_envs=num_envs, **kwargs)
                import torch
                batch_size = torch.Size([num_envs])
            else:
                env = TradingEnv(**kwargs)

        super().__init__(env, device=device, batch_size=batch_size)

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
