"""LOESS Model."""

import numpy as np
import torch

from ..base import ClassicalModel

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    lowess = None


class LOESSModel(ClassicalModel):
    """Locally Estimated Scatterplot Smoothing (LOESS)."""

    def __init__(self, frac=0.66, it=3, **kwargs):
        super().__init__()
        self.frac = frac
        self.it = it
        self.kwargs = kwargs

    def fit(self, X, y):  # noqa: N803
        if lowess is None:
            raise ImportError("statsmodels is required for LOESSModel")

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        if X.ndim == 3:
            X = X.reshape(X.shape[0] * X.shape[1], -1)
            y = y.reshape(y.shape[0] * y.shape[1], -1)

        self.X_train = X
        self.y_train = y
        self._is_fitted = True

    def predict(self, X):  # noqa: N803
        if not self._is_fitted:
            return np.zeros((X.shape[0], 1))

        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = X

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
        )  # pyright: ignore[reportArgumentType]

        out = f(X_np[:, 0])
        if out.ndim == 1:
            out = out[:, np.newaxis]
        return out

    def forward(self, x, **kwargs):
        device = x.device
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            x_np = x_np[:, -1, :]

        out_np = self.predict(x_np)
        return torch.from_numpy(out_np).to(device).to(torch.float32)
