from typing import Any
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from numpy.typing import NDArray


class HierarchicalClusteringAlgorithm:
    """Hierarchical Clustering Algorithm wrapper."""
    
    def __init__(self, n_clusters: int = 2, **kwargs: Any) -> None:
        """Initialize Hierarchical Clustering."""
        self.model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)

    def fit(self, X: NDArray[Any]) -> "HierarchicalClusteringAlgorithm":  # noqa: N803
        """Fit the model."""
        self.model.fit(X)
        return self

    def fit_predict(self, X: NDArray[Any]) -> NDArray[np.int_]:  # noqa: N803
        """Fit and predict labels."""
        return self.model.fit_predict(X)

    def predict(self, X: NDArray[Any]) -> NDArray[np.int_]:  # noqa: N803
        """Predict labels (shorthand for fit_predict since inductive prediction is not supported)."""
        # AgglomerativeClustering does not have a predict method for new data
        if hasattr(self.model, "labels_"):
            return self.model.labels_ # type: ignore
        return self.model.fit_predict(X)
