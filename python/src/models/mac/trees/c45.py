"""C4.5 model implementation."""

from typing import Any

from .decision_tree import DecisionTreeModel


class C45Model(DecisionTreeModel):
    """
    C4.5 Algorithm.
    Improved ID3 with support for continuous attributes and pruning.
    """

    def __init__(self, task: str = "classification", **kwargs: Any) -> None:
        """Initialize C45Model."""
        if task == "classification":
            kwargs["criterion"] = "entropy"
        super().__init__(task=task, **kwargs)
