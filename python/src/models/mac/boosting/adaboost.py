"""AdaBoost model implementation."""

from typing import Any

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from ..base import ClassicalModel


class AdaBoostModel(ClassicalModel):
    """AdaBoost model for classification or regression."""

    def __init__(self, task: str = "regression", **kwargs: Any) -> None:
        """Initialize AdaBoostModel."""
        super().__init__()
        if task == "regression":
            self.model = AdaBoostRegressor(**kwargs)
        else:
            self.model = AdaBoostClassifier(**kwargs)
