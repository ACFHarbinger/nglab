"""Conditional Decision Tree Model."""

from .decision_tree import DecisionTreeModel


class ConditionalDecisionTreeModel(DecisionTreeModel):
    """
    Conditional Decision Tree.
    Approximated by requiring a minimum impurity decrease for splits.
    """

    def __init__(self, task="regression", min_impurity_decrease=0.05, **kwargs):
        kwargs["min_impurity_decrease"] = min_impurity_decrease
        super().__init__(task=task, **kwargs)
