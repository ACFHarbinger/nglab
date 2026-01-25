
"""t-SNE algorithm implementation."""

from typing import Any, cast

from numpy.typing import NDArray
from sklearn.manifold import TSNE


class TSNEAlgorithm:
    """t-SNE algorithm wrapper using sklearn."""

    def __init__(self, n_components: int = 2, **kwargs: Any) -> None:
        """Initialize TSNEAlgorithm."""
        self.model = TSNE(n_components=n_components, **kwargs)
        self._X: NDArray[Any] | None = None

    def fit(self, X: NDArray[Any]) -> "TSNEAlgorithm":  # noqa: N803
        """Store data for fit_transform."""
        # TSNE doesn't have a separate fit method in most versions (it's fit_transform or nothing)
        # but we can store X to allow fit_transform later or just fit_transform now.
        self._X = X
        return self

    def transform(self, X: NDArray[Any]) -> NDArray[Any]:  # noqa: N803
        """Perform fit_transform (t-SNE does not support transform on new data)."""
        # TSNE does not support transform on new data.
        # It only supports fit_transform.
        return cast(NDArray[Any], self.model.fit_transform(X))

    def fit_transform(self, X: NDArray[Any]) -> NDArray[Any]:  # noqa: N803
        """Fit the model and transform the data."""
        return cast(NDArray[Any], self.model.fit_transform(X))
