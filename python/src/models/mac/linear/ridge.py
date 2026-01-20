"""Ridge Regression Model."""

from sklearn.linear_model import Ridge

from ..base import ClassicalModel


class RidgeRegressionModel(ClassicalModel):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__()
        self.model = Ridge(alpha=alpha, **kwargs)
