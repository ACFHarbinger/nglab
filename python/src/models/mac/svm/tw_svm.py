"""Twin Support Vector Machine (TWSVM) implementation."""

from typing import Any

import numpy as np
import torch
from sklearn.svm import LinearSVC

from ..base import ClassicalModel


class TWSVMModel(ClassicalModel):
    """
    Twin Support Vector Machine (TWSVM).
    Simplified implementation for binary classification.
    """

    def __init__(
        self, c1: float = 1.0, c2: float = 1.0, epsilon: float = 1e-5, **kwargs: Any
    ) -> None:
        """Initialize TWSVMModel."""
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.epsilon = epsilon
        self.weights1: np.ndarray[Any, Any] | None = None
        self.weights2: np.ndarray[Any, Any] | None = None
        self.bias1: float | None = None
        self.bias2: float | None = None
        self.fallback: LinearSVC | None = None

    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> None:  # noqa: N803
        """Fit the TWSVM model using structural risk minimization."""
        if y is None:
            raise ValueError("TWSVMModel requires y for fitting.")

        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y

        if X_np.ndim == 3:
            X_np = X_np.reshape(X_np.shape[0] * X_np.shape[1], -1)
            y_np = y_np.reshape(y_np.shape[0] * y_np.shape[1], -1)

        target = y_np.ravel()
        classes = np.unique(target)
        if len(classes) != 2:
            self.fallback = LinearSVC(C=self.c1)
            self.fallback.fit(X_np, target)
            self._is_fitted = True
            return

        self.fallback = None

        A = X_np[target == classes[0]]
        B = X_np[target == classes[1]]

        m1 = A.shape[0]
        m2 = B.shape[0]
        e1 = np.ones((m1, 1))
        e2 = np.ones((m2, 1))

        H = np.hstack((A, e1))
        G = np.hstack((B, e2))

        X_full = np.vstack((H, G))
        y_1 = np.vstack((np.zeros((m1, 1)), np.ones((m2, 1))))

        pseudo_inv1 = np.linalg.pinv(
            X_full.T @ X_full + self.epsilon * np.eye(X_full.shape[1])
        )
        z1 = pseudo_inv1 @ X_full.T @ y_1
        self.weights1 = z1[:-1]
        self.bias1 = float(z1[-1])

        X_full2 = np.vstack((G, H))
        y_2 = np.vstack((np.zeros((m2, 1)), np.ones((m1, 1))))

        pseudo_inv2 = np.linalg.pinv(
            X_full2.T @ X_full2 + self.epsilon * np.eye(X_full2.shape[1])
        )
        z2 = pseudo_inv2 @ X_full2.T @ y_2
        self.weights2 = z2[:-1]
        self.bias2 = float(z2[-1])

        self.classes_ = classes
        self._is_fitted = True

    def predict(self, X: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Predict classes by comparing distances to the twin planes."""
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

        if not self._is_fitted:
            return np.zeros((X_np.shape[0], 1))

        if self.fallback:
            return np.asarray(self.fallback.predict(X_np).reshape(-1, 1))

        if (
            self.weights1 is None
            or self.bias1 is None
            or self.weights2 is None
            or self.bias2 is None
        ):
            return np.zeros((X_np.shape[0], 1))

        dist1 = np.abs(X_np @ self.weights1 + self.bias1)
        dist2 = np.abs(X_np @ self.weights2 + self.bias2)

        preds_idx = (dist1 > dist2).astype(int).ravel()
        return np.asarray(self.classes_[preds_idx].reshape(-1, 1))

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass for the Twin SVM model."""
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
