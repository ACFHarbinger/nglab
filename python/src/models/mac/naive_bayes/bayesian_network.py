"""Bayesian Network Model."""

import numpy as np
import torch
from sklearn.naive_bayes import GaussianNB
from ..base import ClassicalModel


class BayesianNetworkModel(ClassicalModel):
    """
    Bayesian Network (BN) / Bayesian Belief Network (BBN).
    Simplified implementation using Gaussian Naive Bayes.
    """

    def __init__(self, structure="naive", **kwargs):
        super().__init__()
        self.model = GaussianNB(**kwargs)

    def fit(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        self.model.fit(X, y.ravel())
        self._is_fitted = True

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        if not self._is_fitted:
            return np.zeros((X.shape[0], 1))

        return self.model.predict(X).reshape(-1, 1)

    def forward(self, x, **kwargs):
        device = x.device
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            x_np = x_np[:, -1, :]

        out_np = self.predict(x_np)
        return torch.from_numpy(out_np).to(device).to(torch.float32)
