"""Polynomial Regression Model."""

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from ..base import ClassicalModel


class PolynomialRegressionModel(ClassicalModel):
    def __init__(self, degree=2, **kwargs):
        super().__init__()
        self.model = Pipeline(
            [
                ("poly_features", PolynomialFeatures(degree=degree)),
                ("linear_regression", LinearRegression(**kwargs)),
            ]
        )
        self._is_fitted = False

    def fit(self, X, y):  # noqa: N803
        super().fit(X, y)
