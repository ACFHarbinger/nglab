"""Lasso Regression Model."""

from sklearn.linear_model import Lasso

from typing import Any

from ..base import ClassicalModel


class LassoRegressionModel(ClassicalModel):
    def __init__(self, alpha: float = 1.0, **kwargs: Any) -> None:
        super().__init__()
        self.model = Lasso(alpha=alpha, **kwargs)
