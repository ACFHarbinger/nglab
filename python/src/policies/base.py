from abc import ABC, abstractmethod
from typing import Any, Dict

class Policy(ABC):
    """
    Abstract Base Class for all trading policies.
    """
    
    def __init__(self, cfg: Dict[str, Any] = None):
        self.cfg = cfg or {}

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """
        Takes an observation and returns an action.
        
        Args:
            observation: The current state of the environment.
            
        Returns:
            The action to take.
        """
        pass

    def __call__(self, observation: Any) -> Any:
        return self.act(observation)

    def reset(self):
        """Optional reset method for stateful policies."""
        pass
