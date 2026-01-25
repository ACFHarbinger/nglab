"""Stacking ensemble model implementation."""

from typing import Any

from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..base import ClassicalModel


class StackingModel(ClassicalModel):
    """
    Stacked Generalization.
    Requires 'estimators' list of (name, estimator) tuples in kwargs,
    or defaults to simple Linear+Tree stack.
    """

    def __init__(self, task: str = "regression", **kwargs: Any) -> None:
        """Initialize StackingModel."""
        super().__init__()

        if "estimators" not in kwargs:
            if task == "regression":
                kwargs["estimators"] = [
                    ("lr", LinearRegression()),
                    ("tree", DecisionTreeRegressor(max_depth=5)),
                ]
                kwargs.setdefault("final_estimator", LinearRegression())
            else:
                kwargs["estimators"] = [
                    ("lr", LogisticRegression()),
                    ("tree", DecisionTreeClassifier(max_depth=5)),
                ]
                kwargs.setdefault("final_estimator", LogisticRegression())

        if task == "regression":
            self.model = StackingRegressor(**kwargs)
        else:
            self.model = StackingClassifier(**kwargs)
