"""
Linear Discriminant Analysis (LDA) Wrapper.

Provides transformation and classification using Fisher's Linear Discriminant.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from typing import Any
import numpy as np

class LDAAlgorithm:
    """
    Linear Discriminant Analysis for dimensionality reduction and classification.
    """

    def __init__(self, n_components: int | None = None, **kwargs: Any) -> None:
        """
        Initialize LDA.

        Args:
            n_components (int, optional): Number of components to keep. Defaults to None.
            **kwargs: Additional arguments for sklearn's LinearDiscriminantAnalysis.
        """
        self.model = LinearDiscriminantAnalysis(n_components=n_components, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDAAlgorithm":  # noqa: N803
        """Fit the LDA model."""
        self.model.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Project data to maximize class separation."""
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:  # noqa: N803
        """Fit to data, then transform it."""
        return self.model.fit_transform(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Predict class labels."""
        return self.model.predict(X)
