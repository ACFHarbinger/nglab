"""XGBoost Model."""

import xgboost as xgb
from ..base import ClassicalModel


class XGBoostModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = xgb.XGBRegressor(**kwargs)
        else:
            self.model = xgb.XGBClassifier(**kwargs)
