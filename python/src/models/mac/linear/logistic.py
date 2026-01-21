"""Logistic Regression Model."""

from typing import Any

from sklearn.linear_model import LogisticRegression

from ..base import ClassicalModel


class LogisticRegressionModel(ClassicalModel):
    """Logistic Regression model."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize LogisticRegressionModel."""
        super().__init__()
        self.model = LogisticRegression(max_iter=1000, **kwargs)
