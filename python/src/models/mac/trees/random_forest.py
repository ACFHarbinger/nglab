"""Random Forest Model."""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ..base import ClassicalModel


class RandomForestModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = RandomForestRegressor(**kwargs)
        else:
            self.model = RandomForestClassifier(**kwargs)
