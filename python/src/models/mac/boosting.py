"""
Advanced boosting models (XGBoost, LightGBM).
"""
import xgboost as xgb
import lightgbm as lgb
from .base import ClassicalModel

class XGBoostModel(ClassicalModel):
    def __init__(self, task='regression', **kwargs):
        super().__init__()
        if task == 'regression':
            self.model = xgb.XGBRegressor(**kwargs)
        else:
            self.model = xgb.XGBClassifier(**kwargs)

class LightGBMModel(ClassicalModel):
    def __init__(self, task='regression', **kwargs):
        super().__init__()
        if task == 'regression':
            self.model = lgb.LGBMRegressor(**kwargs)
        else:
            self.model = lgb.LGBMClassifier(**kwargs)
