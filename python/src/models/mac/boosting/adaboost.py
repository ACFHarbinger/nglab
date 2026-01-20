from typing import Any

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from ..base import ClassicalModel


class AdaBoostModel(ClassicalModel):
    def __init__(self, task: str = "regression", **kwargs: Any) -> None:
        super().__init__()
        if task == "regression":
            self.model = AdaBoostRegressor(**kwargs)
        else:
            self.model = AdaBoostClassifier(**kwargs)
