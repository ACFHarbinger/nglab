from typing import Any
from sklearn.mixture import GaussianMixture
import numpy as np
from numpy.typing import NDArray


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
        return self.model.predict(X)

    def fit_predict(self, X: NDArray[Any]) -> NDArray[np.int_]:  # noqa: N803
        """Fit and predict labels."""
        return self.model.fit_predict(X)

    def predict_proba(self, X: NDArray[Any]) -> NDArray[np.float64]:  # noqa: N803
        """Predict probabilities."""
        return self.model.predict_proba(X)
