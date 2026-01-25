
"""Principal Component Analysis (PCA) algorithm implementation."""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA


class PCAAlgorithm:
    """Principal Component Analysis (PCA) Algorithm wrapper."""

    def __init__(
        self, n_components: int | float | str | None = None, **kwargs: Any
    ) -> None:
        """Initialize PCA."""
        self.model = PCA(n_components=n_components, **kwargs)

    def fit(self, X: NDArray[Any]) -> "PCAAlgorithm":  # noqa: N803
        """Fit the model."""
        self.model.fit(X)
        return self

    def transform(self, X: NDArray[Any]) -> NDArray[np.float64]:  # noqa: N803
        """Apply dimensionality reduction."""
        return cast(NDArray[np.float64], self.model.transform(X))

    def fit_transform(self, X: NDArray[Any]) -> NDArray[np.float64]:  # noqa: N803
        """Fit and transform."""
        return cast(NDArray[np.float64], self.model.fit_transform(X))

    def inverse_transform(self, X: NDArray[Any]) -> NDArray[np.float64]:  # noqa: N803
        """Transform data back to its original space."""
        return cast(NDArray[np.float64], self.model.inverse_transform(X))
