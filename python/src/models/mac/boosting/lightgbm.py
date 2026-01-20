"""LightGBM Model."""

import lightgbm as lgb

from ..base import ClassicalModel


class LightGBMModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = lgb.LGBMRegressor(**kwargs)
        else:
            self.model = lgb.LGBMClassifier(**kwargs)
