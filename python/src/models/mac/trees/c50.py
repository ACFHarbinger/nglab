"""C5.0 model implementation."""

from typing import Any

from .decision_tree import DecisionTreeModel


class C50Model(DecisionTreeModel):
    """
    C5.0 Algorithm.
    Proprietary improvement over C4.5 (faster, smaller trees).
    """

    def __init__(self, task: str = "classification", **kwargs: Any) -> None:
        """Initialize C50Model."""
        if task == "classification":
            kwargs["criterion"] = "entropy"
        super().__init__(task=task, **kwargs)
