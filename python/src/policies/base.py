"""
Base Policy Definitions for NGLab.

Defines the abstract interface for all trading policies, ensuring consistency
between heuristic and neural approaches.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

class Policy(ABC):
    """
    Abstract Base Class for all trading policies.
    """
    
    def __init__(self, cfg: Dict[str, Any] = None):
        """
        Initialize the policy.

        Args:
            cfg (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
        """
        self.cfg = cfg or {}

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """
        Takes an observation and returns an action.
        
        Args:
            observation (Any): The current state of the environment.
            
        Returns:
            Any: The action to take (e.g., Hold/Buy/Sell).
        """
        pass

    def __call__(self, observation: Any) -> Any:
        """
        Allows calling the policy object directly.
        """
        return self.act(observation)

    def reset(self):
        """
        Optional reset method for stateful policies.
        """
        pass
