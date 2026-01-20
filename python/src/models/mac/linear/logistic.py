"""Logistic Regression Model."""

from typing import Any

from sklearn.linear_model import LogisticRegression

from ..base import ClassicalModel


class LogisticRegressionModel(ClassicalModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.model = LogisticRegression(max_iter=1000, **kwargs)
