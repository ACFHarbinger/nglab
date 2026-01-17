"""
Advanced boosting models (XGBoost, LightGBM).
"""

import lightgbm as lgb
import xgboost as xgb

from .base import ClassicalModel
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)


class XGBoostModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = xgb.XGBRegressor(**kwargs)
        else:
            self.model = xgb.XGBClassifier(**kwargs)


class LightGBMModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = lgb.LGBMRegressor(**kwargs)
        else:
            self.model = lgb.LGBMClassifier(**kwargs)


class GradientBoostingModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = GradientBoostingRegressor(**kwargs)
        else:
            self.model = GradientBoostingClassifier(**kwargs)


class GBRTModel(GradientBoostingModel):
    """Gradient Boosted Regression Trees - Alias for GradientBoostingModel."""
    pass


class AdaBoostModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = AdaBoostRegressor(**kwargs)
        else:
            self.model = AdaBoostClassifier(**kwargs)
