"""Stepwise Regression Model."""

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from ..base import ClassicalModel


class StepwiseRegressionModel(ClassicalModel):
    """Stepwise Regression using Sequential Feature Selection."""
    def __init__(self, direction="forward", n_features_to_select="auto", **kwargs):
        super().__init__()
        self.base_estimator = LinearRegression()
        self.model = SequentialFeatureSelector(
            self.base_estimator,
            n_features_to_select=n_features_to_select,
            direction=direction,
            **kwargs
        )

    def fit(self, X, y):
        super().fit(X, y)
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        if X.ndim == 3:
            X = X.reshape(X.shape[0] * X.shape[1], -1)
            y = y.reshape(y.shape[0] * y.shape[1], -1)
            
        self.selected_features_ = self.model.get_support()
        self.final_model = LinearRegression()
        self.final_model.fit(X[:, self.selected_features_], y)
        self._is_fitted = True

    def forward(self, x, **kwargs):
        if not self._is_fitted:
            return super().forward(x, **kwargs)
        
        device = x.device
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            b, s, f = x_np.shape
            x_np = x_np[:, -1, :]
            
        out_np = self.final_model.predict(x_np[:, self.selected_features_])
        if out_np.ndim == 1:
            out_np = out_np[:, np.newaxis]
            
        return torch.from_numpy(out_np).to(device).to(torch.float32)

    def predict(self, X):
        if not self._is_fitted:
            return np.zeros((X.shape[0], 1))
        return self.final_model.predict(X[:, self.selected_features_])
