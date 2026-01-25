"""Random Forest model suite."""

from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..base import ClassicalModel


class RandomForestModel(ClassicalModel):
    """
    Random Forest wrapper for classification or regression.
    """

    def __init__(self, task: str = "regression", **kwargs: Any) -> None:
        """
        Initialize the Random Forest model.

        Args:
            task (str, optional): 'regression' or 'classification'. Defaults to "regression".
            **kwargs: Additional arguments passed to the underlying sklearn model.
        """
        super().__init__()
        if task == "regression":
            self.model = RandomForestRegressor(**kwargs)
        else:
            self.model = RandomForestClassifier(**kwargs)
