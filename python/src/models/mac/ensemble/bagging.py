"""Bagging ensemble model implementation."""

from typing import Any

from sklearn.ensemble import BaggingClassifier, BaggingRegressor

from ..base import ClassicalModel


class BaggingModel(ClassicalModel):
    """Bagging ensemble for classification or regression."""

    def __init__(self, task: str = "regression", **kwargs: Any) -> None:
        """Initialize BaggingModel."""
        super().__init__()
        if task == "regression":
            self.model = BaggingRegressor(**kwargs)
        else:
            self.model = BaggingClassifier(**kwargs)
