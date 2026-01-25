"""ID3 model implementation."""

from typing import Any

from .decision_tree import DecisionTreeModel


class ID3Model(DecisionTreeModel):
    """
    Iterative Dichotomiser 3 (ID3).
    Approximated using DecisionTreeClassifier with criterion='entropy'.
    """

    def __init__(self, task: str = "classification", **kwargs: Any) -> None:
        """Initialize ID3Model."""
        if task == "regression":
            super().__init__(task=task, **kwargs)
        else:
            kwargs["criterion"] = "entropy"
            super().__init__(task=task, **kwargs)
