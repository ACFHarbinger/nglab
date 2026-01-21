"""LOESS Model."""

from typing import Any

import numpy as np
import torch

from ..base import ClassicalModel

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    lowess = None


class LOESSModel(ClassicalModel):
    """Locally Estimated Scatterplot Smoothing (LOESS)."""

    def __init__(self, frac: float = 0.66, it: int = 3, **kwargs: Any) -> None:
        """Initialize LOESSModel."""
        super().__init__()
        self.frac = frac
        self.it = it
        self.kwargs = kwargs
        self.X_train: np.ndarray[Any, Any] | None = None
        self.y_train: np.ndarray[Any, Any] | None = None

    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> None:  # noqa: N803
        """Fit the LOESS model."""
        if y is None:
            raise ValueError("LOESSModel requires y for fitting.")
        if lowess is None:
            raise ImportError("statsmodels is required for LOESSModel")

        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y

        if X_np.ndim == 3:
            X_np = X_np.reshape(X_np.shape[0] * X_np.shape[1], -1)
            if y_np is not None:
                y_np = y_np.reshape(y_np.shape[0] * y_np.shape[1], -1)

        self.X_train = X_np
        self.y_train = y_np
        self._is_fitted = True

    def predict(self, X: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:  # noqa: N803
        """Predict using the fitted LOESS model."""
        if not self._is_fitted:
            return np.zeros((X.shape[0], 1))

        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = X

        if self.X_train is None or self.y_train is None:
            return np.zeros((X_np.shape[0], 1))

        x_axis = self.X_train[:, 0]
        y_axis = self.y_train.ravel()

        indices = np.argsort(x_axis)

        if lowess is None:
            raise ImportError("statsmodels is required for LOESSModel")

        res = lowess(
            y_axis[indices], x_axis[indices], frac=self.frac, it=self.it, **self.kwargs
        )

        from scipy.interpolate import interp1d

        f = interp1d(
            res[:, 0],
            res[:, 1],
            bounds_error=False,
            fill_value="extrapolate",
        )

        out = np.asarray(f(X_np[:, 0]))
        if out.ndim == 1:
            out = out[:, np.newaxis]
        return out

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass using LOESS prediction."""
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
