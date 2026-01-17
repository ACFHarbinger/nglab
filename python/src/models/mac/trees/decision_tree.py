"""Decision Tree Model."""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..base import ClassicalModel


class DecisionTreeModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = DecisionTreeRegressor(**kwargs)
        else:
            self.model = DecisionTreeClassifier(**kwargs)
