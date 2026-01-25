from __future__ import annotations

from typing import Any

from python.src.utils.registry import POLICY_REGISTRY


class PolicyFactory:
    """Factory for creating policy instances."""

    @staticmethod
    def get_policy(policy_name: str, **kwargs: Any) -> Any:
        """
        Get policy by name from registry.
        
        Args:
            policy_name: Name of the policy to instantiate.
            **kwargs: Arguments to pass to the policy constructor.
            
        Returns:
            An instance of the requested policy.
            
        Raises:
            ModelNotFoundError: If the policy name is not in the registry.
        """
        from ..exceptions import ModelNotFoundError
        
        try:
            policy_cls = POLICY_REGISTRY.get(policy_name.lower())
            return policy_cls(**kwargs)
        except ValueError as e:
            # Re-raise as ModelNotFoundError for consistency in our API
            raise ModelNotFoundError(str(e)) from e
