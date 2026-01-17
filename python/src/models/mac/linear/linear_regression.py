"""Linear Regression Model."""

from sklearn.linear_model import LinearRegression
from ..base import ClassicalModel


class LinearRegressionModel(ClassicalModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LinearRegression(**kwargs)
