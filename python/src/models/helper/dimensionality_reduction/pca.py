from typing import Any, Optional, Union
from sklearn.decomposition import PCA
import numpy as np
from numpy.typing import NDArray


class PCAAlgorithm:
    """Principal Component Analysis (PCA) Algorithm wrapper."""
    
    def __init__(self, n_components: Optional[Union[int, float, str]] = None, **kwargs: Any) -> None:
        """Initialize PCA."""
        self.model = PCA(n_components=n_components, **kwargs)

    def fit(self, X: NDArray[Any]) -> "PCAAlgorithm":  # noqa: N803
        """Fit the model."""
        self.model.fit(X)
        return self

    def transform(self, X: NDArray[Any]) -> NDArray[np.float64]:  # noqa: N803
        """Apply dimensionality reduction."""
        return self.model.transform(X)

    def fit_transform(self, X: NDArray[Any]) -> NDArray[np.float64]:  # noqa: N803
        """Fit and transform."""
        return self.model.fit_transform(X)

    def inverse_transform(self, X: NDArray[Any]) -> NDArray[np.float64]:  # noqa: N803
        """Transform data back to its original space."""
        return self.model.inverse_transform(X)
