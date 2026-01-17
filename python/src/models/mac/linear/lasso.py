"""Lasso Regression Model."""

from sklearn.linear_model import Lasso
from ..base import ClassicalModel


class LassoRegressionModel(ClassicalModel):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__()
        self.model = Lasso(alpha=alpha, **kwargs)
