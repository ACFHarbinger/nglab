"""Linear Regression Model."""

from typing import Any

from sklearn.linear_model import LinearRegression

from ..base import ClassicalModel


class LinearRegressionModel(ClassicalModel):
    """
    Linear Regression wrapper.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Linear Regression model.

        Args:
            **kwargs: Additional arguments passed to the underlying sklearn model.
        """
        super().__init__()
        self.model = LinearRegression(**kwargs)
