"""
Tree-based models.
"""

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base import ClassicalModel


class DecisionTreeModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = DecisionTreeRegressor(**kwargs)
        else:
            self.model = DecisionTreeClassifier(**kwargs)


class RandomForestModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = RandomForestRegressor(**kwargs)
        else:
            self.model = RandomForestClassifier(**kwargs)


class GradientBoostingModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = GradientBoostingRegressor(**kwargs)
        else:
            self.model = GradientBoostingClassifier(**kwargs)
