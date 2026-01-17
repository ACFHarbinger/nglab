"""Decision Stump Model."""

from .decision_tree import DecisionTreeModel


class DecisionStumpModel(DecisionTreeModel):
    """Decision Stump: A Decision Tree with max_depth=1."""
    def __init__(self, task="regression", **kwargs):
        kwargs["max_depth"] = 1
        super().__init__(task=task, **kwargs)
