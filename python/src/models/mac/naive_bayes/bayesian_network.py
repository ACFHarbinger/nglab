"""Bayesian Network model implementation."""

from typing import Any

import numpy as np
import torch
from sklearn.naive_bayes import GaussianNB

from ..base import ClassicalModel


class BayesianNetworkModel(ClassicalModel):
    """
    Bayesian Network (BN) / Bayesian Belief Network (BBN).
    Simplified implementation using Gaussian Naive Bayes.
    """

    def __init__(self, structure: str = "naive", **kwargs: Any) -> None:
        """Initialize BayesianNetworkModel."""
        super().__init__()
        self.model = GaussianNB(**kwargs)

    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> None:  # noqa: N803
        """Fit the Bayesian Network model."""
        if y is None:
            raise ValueError("BayesianNetworkModel requires y for fitting.")

        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        self.model.fit(X_np, y_np.ravel())
        self._is_fitted = True

    def predict(self, X: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Predict labels using the fitted Bayesian Network."""
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

        if not self._is_fitted:
            return np.zeros((X_np.shape[0], 1))

        out_np = np.asarray(self.model.predict(X_np))
        return out_np.reshape(-1, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass for the Bayesian Network model."""
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
