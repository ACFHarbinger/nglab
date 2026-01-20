from typing import Any

import lightgbm as lgb

from ..base import ClassicalModel


class LightGBMModel(ClassicalModel):
    def __init__(self, task: str = "regression", **kwargs: Any) -> None:
        super().__init__()
        if task == "regression":
            self.model = lgb.LGBMRegressor(**kwargs)
        else:
            self.model = lgb.LGBMClassifier(**kwargs)
