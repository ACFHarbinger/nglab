from typing import Any

from sklearn.ensemble import BaggingClassifier, BaggingRegressor

from ..base import ClassicalModel


class BaggingModel(ClassicalModel):
    def __init__(self, task: str = "regression", **kwargs: Any) -> None:
        super().__init__()
        if task == "regression":
            self.model = BaggingRegressor(**kwargs)
        else:
            self.model = BaggingClassifier(**kwargs)
