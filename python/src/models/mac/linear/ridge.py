"""Ridge Regression Model."""

from typing import Any

from sklearn.linear_model import Ridge

from ..base import ClassicalModel


class RidgeRegressionModel(ClassicalModel):
    def __init__(self, alpha: float = 1.0, **kwargs: Any) -> None:
        super().__init__()
        self.model = Ridge(alpha=alpha, **kwargs)
