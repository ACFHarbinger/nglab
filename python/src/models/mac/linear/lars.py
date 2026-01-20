"""LARS Model."""

from sklearn.linear_model import Lars

from typing import Any

from ..base import ClassicalModel


class LARSModel(ClassicalModel):
    def __init__(self, n_nonzero_coefs: int = 500, **kwargs: Any) -> None:
        super().__init__()
        self.model = Lars(n_nonzero_coefs=n_nonzero_coefs, **kwargs)
