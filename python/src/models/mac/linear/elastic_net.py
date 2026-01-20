"""Elastic Net Model."""

from sklearn.linear_model import ElasticNet

from ..base import ClassicalModel


class ElasticNetModel(ClassicalModel):
    def __init__(self, alpha=1.0, l1_ratio=0.5, **kwargs):
        super().__init__()
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs)
