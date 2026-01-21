"""Lasso Regression Model."""

from typing import Any

from sklearn.linear_model import Lasso

from ..base import ClassicalModel


class LassoRegressionModel(ClassicalModel):
    """Lasso Regression (L1 regularization)."""

    def __init__(self, alpha: float = 1.0, **kwargs: Any) -> None:
        """Initialize LassoRegressionModel."""
        super().__init__()
        self.model = Lasso(alpha=alpha, **kwargs)
