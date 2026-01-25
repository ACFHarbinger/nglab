
"""K-Medians clustering algorithm implementation."""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


class KMediansAlgorithm:
    """K-Medians Clustering Algorithm."""

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        """Initialize K-Medians."""
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: NDArray[Any] | None = None
        self.labels_: NDArray[np.int_] | None = None

    def fit(self, X: NDArray[Any]) -> "KMediansAlgorithm":  # noqa: N803
        """Fit the model."""
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        # Initialize centroids randomly from data points
        random_indices = rng.permutation(n_samples)[: self.n_clusters]
        self.cluster_centers_ = X[random_indices].copy()

        for _i in range(self.max_iter):
            # Assign labels based on L1 distance (Manhattan)
            # dist shape: (n_samples, n_clusters)
            if self.cluster_centers_ is None:
                break
            dist = np.sum(
                np.abs(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]),
                axis=2,
            )
            self.labels_ = cast(NDArray[np.int_], np.argmin(dist, axis=1))

            new_centers = np.empty((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                mask = self.labels_ == k
                if np.any(mask):
                    new_centers[k] = np.median(X[mask], axis=0)
                else:
                    new_centers[k] = self.cluster_centers_[k]

            # Check convergence
            center_shift = float(np.sum(np.abs(new_centers - self.cluster_centers_)))
            if center_shift < self.tol:
                self.cluster_centers_ = new_centers
                break

            self.cluster_centers_ = new_centers

        return self

    def predict(self, X: NDArray[Any]) -> NDArray[np.int_]:  # noqa: N803
        """Predict labels."""
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet.")
        dist = np.sum(
            np.abs(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]),
            axis=2,
        )
        return cast(NDArray[np.int_], np.argmin(dist, axis=1))

    def fit_predict(self, X: NDArray[Any]) -> NDArray[np.int_]:  # noqa: N803
        """Fit and predict labels."""
        self.fit(X)
        if self.labels_ is None:
            raise ValueError("Fitting failed to produce labels.")
        return self.labels_
