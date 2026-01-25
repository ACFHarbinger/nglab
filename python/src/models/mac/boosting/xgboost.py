"""XGBoost model implementation."""

from typing import Any

import xgboost as xgb

from ..base import ClassicalModel


class XGBoostModel(ClassicalModel):
    """
    XGBoost wrapper for classification or regression.
    """

    def __init__(self, task: str = "regression", **kwargs: Any) -> None:
        """
        Initialize the XGBoost model.

        Args:
            task (str, optional): 'regression' or 'classification'. Defaults to "regression".
            **kwargs: Additional arguments passed to the underlying xgboost model.
        """
        super().__init__()
        if task == "regression":
            self.model = xgb.XGBRegressor(**kwargs)
        else:
            self.model = xgb.XGBClassifier(**kwargs)
