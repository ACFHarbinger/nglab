
"""DBSCAN clustering algorithm implementation."""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN


class DBSCANAlgorithm:
    """DBSCAN Clustering Algorithm wrapper."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs: Any) -> None:
        """Initialize DBSCAN."""
        self.model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)

    def fit(self, X: NDArray[Any]) -> "DBSCANAlgorithm":  # noqa: N803
        """Fit the model."""
        self.model.fit(X)
        return self

    def fit_predict(self, X: NDArray[Any]) -> NDArray[np.int_]:  # noqa: N803
        """Fit and predict labels."""
        return cast(NDArray[np.int_], self.model.fit_predict(X))

    def predict(self, X: NDArray[Any]) -> NDArray[np.int_]:  # noqa: N803
        """Predict labels."""
        # DBSCAN sklearn does not have a predict method for new data.
        if hasattr(self.model, "labels_"):
            return cast(NDArray[np.int_], self.model.labels_)
        return cast(NDArray[np.int_], self.model.fit_predict(X))
