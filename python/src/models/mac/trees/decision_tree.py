"""Decision Tree Model."""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..base import ClassicalModel


class DecisionTreeModel(ClassicalModel):
    """
    Decision Tree wrapper for classification or regression.
    """

    def __init__(self, task="regression", **kwargs):
        """
        Initialize the Decision Tree model.

        Args:
            task (str, optional): 'regression' or 'classification'. Defaults to "regression".
            **kwargs: Additional arguments passed to the underlying sklearn model.
        """
        super().__init__()
        if task == "regression":
            self.model = DecisionTreeRegressor(**kwargs)
        else:
            self.model = DecisionTreeClassifier(**kwargs)
