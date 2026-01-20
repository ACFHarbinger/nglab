from typing import Any

from .decision_tree import DecisionTreeModel


class DecisionStumpModel(DecisionTreeModel):
    """Decision Stump: A Decision Tree with max_depth=1."""

    def __init__(self, task: str = "regression", **kwargs: Any) -> None:
        kwargs["max_depth"] = 1
        super().__init__(task=task, **kwargs)
