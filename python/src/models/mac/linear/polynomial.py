"""Polynomial Regression Model."""

from typing import Any

import torch
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from ..base import ClassicalModel


class PolynomialRegressionModel(ClassicalModel):
    """Polynomial Regression model."""

    def __init__(self, degree: int = 2, **kwargs: Any) -> None:
        """Initialize PolynomialRegressionModel."""
        super().__init__()
        self.model = Pipeline(
            [
                ("poly_features", PolynomialFeatures(degree=degree)),
                ("linear_regression", LinearRegression(**kwargs)),
            ]
        )
        self._is_fitted = False

    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> None:  # noqa: N803
        """Fit the polynomial regression model."""
        super().fit(X, y)
