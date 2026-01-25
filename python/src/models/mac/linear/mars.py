"""Multivariate Adaptive Regression Splines (MARS) model implementation."""

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from ..base import ClassicalModel


class MARSModel(ClassicalModel):
    """Simplified Multivariate Adaptive Regression Splines (Piecewise Linear)."""

    def __init__(self, n_segments: int = 5, **kwargs: Any) -> None:
        """Initialize MARSModel."""
        super().__init__()
        self.n_segments = n_segments
        self.model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=1)),
                ("linear", LinearRegression(**kwargs)),
            ]
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> None:  # noqa: N803
        """Fit the MARS model to data."""
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

        hinges = []
        knots = np.array([])
        for i in range(X_np.shape[1]):
            feat = X_np[:, i]
            knots = np.linspace(feat.min(), feat.max(), self.n_segments)
            for knot in knots:
                hinges.append(np.maximum(0, feat - knot))
                hinges.append(np.maximum(0, knot - feat))

        X_hinge = np.column_stack(hinges)
        self.model = LinearRegression()
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y

        if y_np is not None and y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np.ravel()

        self.model.fit(X_hinge, y_np)
        self.knots_ = knots
        self._is_fitted = True

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass using MARS prediction."""
        if not self._is_fitted:
            return super().forward(
                x, return_embedding=return_embedding, return_sequence=return_sequence
            )

        device = x.device
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            x_np = x_np[:, -1, :]

        hinges = []
        for i in range(x_np.shape[1]):
            feat = x_np[:, i]
            knots = np.linspace(feat.min(), feat.max(), self.n_segments)
            for knot in knots:
                hinges.append(np.maximum(0, feat - knot))
                hinges.append(np.maximum(0, knot - feat))

        X_hinge = np.column_stack(hinges)
        out_np = self.model.predict(X_hinge)
        if out_np.ndim == 1:
            out_np = out_np[:, np.newaxis]

        return torch.from_numpy(out_np).to(device).to(torch.float32)

    def predict(self, X: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Predict using the fitted MARS model."""
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

        if not self._is_fitted:
            return np.zeros((X_np.shape[0], 1))

        hinges = []
        for i in range(X_np.shape[1]):
            feat = X_np[:, i]
            knots = np.linspace(feat.min(), feat.max(), self.n_segments)
            for knot in knots:
                hinges.append(np.maximum(0, feat - knot))
                hinges.append(np.maximum(0, knot - feat))

        X_hinge = np.column_stack(hinges)
        return np.asarray(self.model.predict(X_hinge))
