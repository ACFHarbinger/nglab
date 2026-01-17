"""Gradient Boosting Model."""

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from ..base import ClassicalModel


class GradientBoostingModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = GradientBoostingRegressor(**kwargs)
        else:
            self.model = GradientBoostingClassifier(**kwargs)


class GBRTModel(GradientBoostingModel):
    """Gradient Boosted Regression Trees - Alias for GradientBoostingModel."""
    pass
