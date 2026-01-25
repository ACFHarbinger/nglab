
"""Gaussian Mixture Model (GMM) algorithm implementation."""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture


class GMMAlgorithm:
    """Gaussian Mixture Model Algorithm."""

    def __init__(self, n_components: int = 1, **kwargs: Any) -> None:
        """Initialize GMM."""
        self.model = GaussianMixture(n_components=n_components, **kwargs)

    def fit(self, X: NDArray[Any]) -> "GMMAlgorithm":  # noqa: N803
        """Fit the model."""
        self.model.fit(X)
        return self

    def predict(self, X: NDArray[Any]) -> NDArray[np.int_]:  # noqa: N803
        """Predict labels."""
        return cast(NDArray[np.int_], self.model.predict(X))

    def fit_predict(self, X: NDArray[Any]) -> NDArray[np.int_]:  # noqa: N803
        """Fit and predict labels."""
        return cast(NDArray[np.int_], self.model.fit_predict(X))

    def predict_proba(self, X: NDArray[Any]) -> NDArray[np.float64]:  # noqa: N803
        """Predict probabilities."""
        return cast(NDArray[np.float64], self.model.predict_proba(X))
