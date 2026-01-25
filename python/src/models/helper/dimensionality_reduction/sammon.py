
"""Sammon Mapping algorithm implementation."""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA


class SammonMappingAlgorithm:
    """Sammon Mapping Dimensionality Reduction Algorithm."""

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        lr: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize Sammon Mapping."""
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.embedding_: NDArray[np.float64] | None = None

    def fit(self, X: NDArray[Any]) -> "SammonMappingAlgorithm":  # noqa: N803
        """Fit the model (Sammon is transductive, usually calls fit_transform)."""
        return self

    def fit_transform(self, X: NDArray[Any] | torch.Tensor) -> NDArray[np.float64]:
        """Fit and transform."""
        # Expecting numpy input, convert to torch
        if not isinstance(X, torch.Tensor):
            X_t = torch.tensor(X, dtype=torch.float32)
        else:
            X_t = X.float()

        n_samples = X_t.shape[0]

        # 1. Compute pairwise distances in high-dim (no gradient needed for input)
        diff = X_t.unsqueeze(1) - X_t.unsqueeze(0)
        dist_high = torch.norm(diff, dim=-1) + 1e-6  # Avoid div by zero

        # Mask for off-diagonal
        mask = ~torch.eye(n_samples, dtype=torch.bool, device=X_t.device)
        d_star = dist_high[mask]
        c = torch.sum(d_star)

        # 2. Initialize Low-dim Y (using PCA)
        try:
            pca = PCA(n_components=self.n_components)
            Y_init = pca.fit_transform(X_t.cpu().numpy())
            Y = torch.tensor(
                Y_init, dtype=torch.float32, requires_grad=True, device=X_t.device
            )
        except Exception:
            Y = torch.randn(
                n_samples, self.n_components, requires_grad=True, device=X_t.device
            )

        optimizer = torch.optim.Adam([Y], lr=self.lr)

        for _i in range(self.max_iter):
            optimizer.zero_grad()

            # Distance in low-dim
            diff_y = Y.unsqueeze(1) - Y.unsqueeze(0)
            dist_low = torch.norm(diff_y, dim=-1) + 1e-6

            d = dist_low[mask]

            # Sammon Stress: E = (1/c) * sum( (d* - d)^2 / d* )
            loss = (1.0 / c) * torch.sum(((d_star - d) ** 2) / d_star)

            loss.backward()
            optimizer.step()

            if loss.item() < self.tol:
                break

        self.embedding_ = Y.detach().cpu().numpy()
        return self.embedding_

    def transform(self, X: NDArray[Any]) -> NDArray[np.float64]:  # noqa: N803
        """Transform (for Sammon, this often returns the stored embedding or re-fits)."""
        if self.embedding_ is not None and X.shape[0] == self.embedding_.shape[0]:
            return self.embedding_
        return self.fit_transform(X)
