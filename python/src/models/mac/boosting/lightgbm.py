"""LightGBM model implementation."""

from typing import Any

import lightgbm as lgb

from ..base import ClassicalModel


class LightGBMModel(ClassicalModel):
    """LightGBM model for classification or regression."""

    def __init__(self, task: str = "regression", **kwargs: Any) -> None:
        """Initialize LightGBMModel."""
        super().__init__()
        if task == "regression":
            self.model = lgb.LGBMRegressor(**kwargs)
        else:
            self.model = lgb.LGBMClassifier(**kwargs)
