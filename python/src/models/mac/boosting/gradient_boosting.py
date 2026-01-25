"""Gradient Boosting model implementation."""

from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from ..base import ClassicalModel


class GradientBoostingModel(ClassicalModel):
    """Gradient Boosting model for classification or regression."""

    def __init__(self, task: str = "regression", **kwargs: Any) -> None:
        """Initialize GradientBoostingModel."""
        super().__init__()
        if task == "regression":
            self.model = GradientBoostingRegressor(**kwargs)
        else:
            self.model = GradientBoostingClassifier(**kwargs)


class GBRTModel(GradientBoostingModel):
    """Gradient Boosted Regression Trees - Alias for GradientBoostingModel."""

    pass
