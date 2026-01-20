from typing import Any
from sklearn.cluster import KMeans
import numpy as np
from numpy.typing import NDArray


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
        return self.model.predict(X)

    def fit_predict(self, X: NDArray[Any]) -> NDArray[np.int_]:  # noqa: N803
        """Fit and predict labels."""
        return self.model.fit_predict(X)
