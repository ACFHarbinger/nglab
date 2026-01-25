"""Stepwise Regression model implementation."""

from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt  # Use this for modern numpy hints
import torch
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

from ..base import ClassicalModel


class StepwiseRegressionModel(ClassicalModel):
    """Stepwise Regression using Sequential Feature Selection."""

    def __init__(
        self,
        direction: str = "forward",
        n_features_to_select: str | int | float = "auto",
        **kwargs: Any,
    ) -> None:
        """Initialize StepwiseRegressionModel."""
        super().__init__()
        self.base_estimator = LinearRegression()

        # Handle literals for mypy/basedpyright
        sel_direction = cast(Literal["forward", "backward"], direction)

        # SequentialFeatureSelector expects specific types for n_features_to_select
        self.model = SequentialFeatureSelector(
            self.base_estimator,
            n_features_to_select=n_features_to_select,
            direction=sel_direction,
            **kwargs,
        )
        # Fix: np.ndarray takes 1 type arg (dtype), not 2
        self.selected_features_: npt.NDArray[Any] | None = None
        self.final_model: LinearRegression | None = None

    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> None:  # noqa: N803
        """Fit the stepwise regression model."""
        if y is None:
            raise ValueError("StepwiseRegressionModel requires y for fitting.")

        # Ensure we have numpy arrays for sklearn
        X_np: npt.NDArray[Any] = (
            X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        )
        y_np: npt.NDArray[Any] = (
            y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        )

        if X_np.ndim == 3:
            X_np = X_np.reshape(X_np.shape[0] * X_np.shape[1], -1)
            y_np = y_np.reshape(y_np.shape[0] * y_np.shape[1], -1)

        self.model.fit(X_np, y_np)

        support = self.model.get_support()
        # Fix: Explicitly cast the support mask
        self.selected_features_ = np.asarray(support, dtype=bool)

        self.final_model = LinearRegression()
        # Use boolean indexing safely
        self.final_model.fit(X_np[:, self.selected_features_], y_np)
        self._is_fitted = True

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass using selected features."""
        if not self._is_fitted:
            return super().forward(
                x, return_embedding=return_embedding, return_sequence=return_sequence
            )

        device = x.device
        x_np: npt.NDArray[Any] = x.detach().cpu().numpy()

        # Handle time-series indexing
        if x_np.ndim == 3:
            x_np = x_np[:, -1, :]

        if self.selected_features_ is None or self.final_model is None:
            batch_size = x_np.shape[0]
            out_np = np.zeros((batch_size, 1), dtype=np.float32)
        else:
            # Fix: Ensure predict output is typed correctly
            preds = self.final_model.predict(x_np[:, self.selected_features_])
            out_np = np.asarray(preds, dtype=np.float32)
            if out_np.ndim == 1:
                out_np = out_np[:, np.newaxis]

        return torch.from_numpy(out_np).to(device).to(torch.float32)

    # Fix: Corrected Union typing and Return type
    def predict(self, X: npt.NDArray[Any] | torch.Tensor) -> npt.NDArray[Any]:
        """Predict using the fitted model."""
        X_np: npt.NDArray[Any] = (
            X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        )

        if (
            not self._is_fitted
            or self.selected_features_ is None
            or self.final_model is None
        ):
            return np.zeros((X_np.shape[0], 1))

        preds = self.final_model.predict(X_np[:, self.selected_features_])
        return np.asarray(preds)
