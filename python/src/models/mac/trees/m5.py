"""M5 model tree implementation."""

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from ..base import ClassicalModel


class M5Model(ClassicalModel):
    """
    M5 Algorithm for Regression.
    Builds a tree and then fits linear regression models at the leaves.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize M5Model."""
        super().__init__()
        self.tree = DecisionTreeRegressor(**kwargs)
        self.leaf_models: dict[int, LinearRegression | None] = {}

    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> None:  # noqa: N803
        """Fit the M5 model tree."""
        if y is None:
            raise ValueError("M5Model requires y for fitting.")

        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y

        if X_np.ndim == 3:
            X_np = X_np.reshape(X_np.shape[0] * X_np.shape[1], -1)

            if y_np.ndim > 1:
                y_np = y_np.reshape(y_np.shape[0] * y_np.shape[1], -1)

        self.tree.fit(X_np, y_np)

        leaves = np.asarray(self.tree.apply(X_np))
        unique_leaves = np.unique(leaves)

        for leaf in unique_leaves:
            mask = leaves == leaf
            X_leaf = X_np[mask]
            y_leaf = y_np[mask]

            if len(X_leaf) > X_np.shape[1] + 1:
                model = LinearRegression()
                model.fit(X_leaf, y_leaf)
                self.leaf_models[int(leaf)] = model
            else:
                self.leaf_models[int(leaf)] = None

        self._is_fitted = True

    def predict(self, X: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Predict values using the fitted model tree and leaf-level linear models."""
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

        if not self._is_fitted:
            return np.zeros((X_np.shape[0], 1))

        leaves = np.asarray(self.tree.apply(X_np))
        pred = np.zeros(
            (X_np.shape[0], 1)
            if self.tree.n_outputs_ == 1
            else (X_np.shape[0], self.tree.n_outputs_)
        )
        tree_pred = np.asarray(self.tree.predict(X_np))
        if tree_pred.ndim == 1:
            tree_pred = tree_pred[:, np.newaxis]

        for i, leaf in enumerate(leaves):
            lm = self.leaf_models.get(int(leaf))
            if lm:
                p = np.asarray(lm.predict(X_np[i : i + 1]))
                pred[i] = p if p.ndim == 1 else p[0]
            else:
                pred[i] = tree_pred[i]

        return pred

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass for the M5 model."""
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
