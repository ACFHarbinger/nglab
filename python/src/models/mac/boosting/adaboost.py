"""AdaBoost Model."""

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from ..base import ClassicalModel


class AdaBoostModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = AdaBoostRegressor(**kwargs)
        else:
            self.model = AdaBoostClassifier(**kwargs)
