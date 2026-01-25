
"""K-Means clustering algorithm implementation."""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans


class KMeansAlgorithm:
    """K-Means Clustering Algorithm."""

    def __init__(self, n_clusters: int = 8, **kwargs: Any) -> None:
        """Initialize K-Means."""
        self.model = KMeans(n_clusters=n_clusters, **kwargs)

    def fit(self, X: NDArray[Any]) -> "KMeansAlgorithm":  # noqa: N803
        """Fit the model."""
        self.model.fit(X)
        return self

    def predict(self, X: NDArray[Any]) -> NDArray[np.int_]:  # noqa: N803
        """Predict labels."""
        return cast(NDArray[np.int_], self.model.predict(X))

    def fit_predict(self, X: NDArray[Any]) -> NDArray[np.int_]:  # noqa: N803
        """Fit and predict labels."""
        return cast(NDArray[np.int_], self.model.fit_predict(X))
