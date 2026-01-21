"""LARS Model."""

from typing import Any

from sklearn.linear_model import Lars

from ..base import ClassicalModel


class LARSModel(ClassicalModel):
    """Least Angle Regression (LARS) model."""

    def __init__(self, n_nonzero_coefs: int = 500, **kwargs: Any) -> None:
        """Initialize LARSModel."""
        super().__init__()
        self.model = Lars(n_nonzero_coefs=n_nonzero_coefs, **kwargs)
