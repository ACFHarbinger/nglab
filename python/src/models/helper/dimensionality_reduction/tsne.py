from sklearn.manifold import TSNE


from typing import Any
import numpy as np

class TSNEAlgorithm:
    def __init__(self, n_components: int = 2, **kwargs: Any) -> None:
        self.model = TSNE(n_components=n_components, **kwargs)

    def fit(self, X: np.ndarray) -> "TSNEAlgorithm":  # noqa: N803
        # TSNE doesn't have a separate fit method in most versions (it's fit_transform or nothing)
        # but we can store X to allow fit_transform later or just fit_transform now.
        self._X = X
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        # TSNE does not support transform on new data.
        # It only supports fit_transform.
        return self.model.fit_transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return self.model.fit_transform(X)
