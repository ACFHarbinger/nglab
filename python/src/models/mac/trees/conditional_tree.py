"""Conditional Decision Tree model implementation."""

from typing import Any

from .decision_tree import DecisionTreeModel


class ConditionalDecisionTreeModel(DecisionTreeModel):
    """
    Conditional Decision Tree.
    Approximated by requiring a minimum impurity decrease for splits.
    """

    def __init__(
        self,
        task: str = "regression",
        min_impurity_decrease: float = 0.05,
        **kwargs: Any,
    ) -> None:
        """Initialize ConditionalDecisionTreeModel."""
        kwargs["min_impurity_decrease"] = min_impurity_decrease
        super().__init__(task=task, **kwargs)
