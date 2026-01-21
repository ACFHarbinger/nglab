"""Ridge Regression Model."""

from typing import Any

from sklearn.linear_model import Ridge

from ..base import ClassicalModel


class RidgeRegressionModel(ClassicalModel):
    """Ridge Regression (L2 regularization)."""

    def __init__(self, alpha: float = 1.0, **kwargs: Any) -> None:
        """Initialize RidgeRegressionModel."""
        super().__init__()
        self.model = Ridge(alpha=alpha, **kwargs)
