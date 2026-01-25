"""Voting ensemble model implementation."""

from typing import Any

from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..base import ClassicalModel


class VotingModel(ClassicalModel):
    """Voting Ensemble (Soft/Hard Voting or Weighted Average)."""

    def __init__(self, task: str = "regression", **kwargs: Any) -> None:
        """Initialize VotingModel."""
        super().__init__()

        if "estimators" not in kwargs:
            if task == "regression":
                kwargs["estimators"] = [
                    ("lr", LinearRegression()),
                    ("tree", DecisionTreeRegressor(max_depth=5)),
                ]
            else:
                kwargs["estimators"] = [
                    ("lr", LogisticRegression()),
                    ("tree", DecisionTreeClassifier(max_depth=5)),
                ]

        if task == "regression":
            self.model = VotingRegressor(**kwargs)
        else:
            self.model = VotingClassifier(**kwargs)


class WeightedAverageModel(VotingModel):
    """Weighted Average (Blending). Alias for VotingModel."""

    pass
