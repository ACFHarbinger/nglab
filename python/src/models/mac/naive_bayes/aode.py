"""Averaged One-Dependence Estimators (AODE) model implementation."""

from typing import Any

import numpy as np
import torch
from sklearn.naive_bayes import GaussianNB

from ..base import ClassicalModel


class AODEModel(ClassicalModel):
    """
    Averaged One-Dependence Estimators (AODE).
    Approximated by averaging multiple Naive Bayes models.
    """

    def __init__(self, n_estimators: int = 10, **kwargs: Any) -> None:
        """Initialize AODEModel."""
        super().__init__()
        self.n_estimators = n_estimators
        self.models: list[GaussianNB] = []
        self.feature_subsets: list[list[int]] = []

    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> None:  # noqa: N803
        """Fit the AODE model."""
        if y is None:
            raise ValueError("AODEModel requires y for fitting.")

        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y

        n_features = X_np.shape[1]
        self.models = []
        self.feature_subsets = []

        candidates = list(range(n_features))
        if n_features > self.n_estimators:
            candidates = list(
                np.random.choice(candidates, self.n_estimators, replace=False)
            )

        for _ in candidates:
            clf = GaussianNB()
            clf.fit(X_np, y_np.ravel())
            self.models.append(clf)

        self._is_fitted = True

    def predict(self, X: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Predict labels by averaging multiple Naive Bayes models."""
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

        if not self._is_fitted:
            return np.zeros((X_np.shape[0], 1))

        preds = []
        for model in self.models:
            preds.append(model.predict_proba(X_np))

        avg_proba = np.mean(preds, axis=0)
        return np.asarray(np.argmax(avg_proba, axis=1).reshape(-1, 1))

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass for the AODE model."""
        if not self._is_fitted:
            return super().forward(
                x, return_embedding=return_embedding, return_sequence=return_sequence
            )

        device = x.device
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            x_np = x_np[:, -1, :]

        out_np = self.predict(x_np)
        return torch.from_numpy(out_np).to(device).to(torch.float32)
