"""Bagging Model."""

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from ..base import ClassicalModel


class BaggingModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = BaggingRegressor(**kwargs)
        else:
            self.model = BaggingClassifier(**kwargs)
