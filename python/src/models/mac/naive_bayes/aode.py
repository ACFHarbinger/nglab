"""AODE Model."""

import numpy as np
import torch
from sklearn.naive_bayes import GaussianNB

from ..base import ClassicalModel


class AODEModel(ClassicalModel):
    """
    Averaged One-Dependence Estimators (AODE).
    Approximated by averaging multiple Naive Bayes models.
    """

    def __init__(self, n_estimators=10, **kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.models = []
        self.feature_subsets = []

    def fit(self, X, y):  # noqa: N803
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        n_features = X.shape[1]
        self.models = []
        self.feature_subsets = []

        candidates = list(range(n_features))
        if n_features > self.n_estimators:
            candidates = np.random.choice(candidates, self.n_estimators, replace=False)

        for _ in candidates:
            clf = GaussianNB()
            clf.fit(X, y.ravel())
            self.models.append(clf)

        self._is_fitted = True

    def predict(self, X):  # noqa: N803
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        if not self._is_fitted:
            return np.zeros((X.shape[0], 1))

        preds = []
        for model in self.models:
            preds.append(model.predict_proba(X))

        avg_proba = np.mean(preds, axis=0)
        return np.argmax(avg_proba, axis=1).reshape(-1, 1)

    def forward(self, x, **kwargs):
        device = x.device
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            x_np = x_np[:, -1, :]

        out_np = self.predict(x_np)
        return torch.from_numpy(out_np).to(device).to(torch.float32)
