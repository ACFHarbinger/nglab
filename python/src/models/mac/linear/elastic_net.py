"""Elastic Net Model."""

from typing import Any

from sklearn.linear_model import ElasticNet

from ..base import ClassicalModel


class ElasticNetModel(ClassicalModel):
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, **kwargs: Any) -> None:
        super().__init__()
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs)
