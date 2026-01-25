from __future__ import annotations

from typing import Any

from python.src.utils.registry import PIPELINE_REGISTRY


class PipelineFactory:
    """Factory for creating pipeline components (Trainers, modules, etc.)."""

    @staticmethod
    def get_pipeline(name: str, **kwargs: Any) -> Any:
        """
        Get pipeline component by name.
        
        Args:
            name: Name of the pipeline component to instantiate.
            **kwargs: Arguments to pass to the constructor.
            
        Returns:
            An instance of the requested pipeline component.
            
        Raises:
            ModelNotFoundError: If the component name is not in the registry.
        """
        from ..exceptions import ModelNotFoundError
        
        try:
            pipeline_cls = PIPELINE_REGISTRY.get(name.lower())
            return pipeline_cls(**kwargs)
        except ValueError as e:
            raise ModelNotFoundError(str(e)) from e
