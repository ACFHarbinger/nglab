from typing import Any, Optional, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.manifold import TSNE


class TSNEAlgorithm:
    def __init__(self, n_components: int = 2, **kwargs: Any) -> None:
        self.model = TSNE(n_components=n_components, **kwargs)
        self._X: Optional[NDArray[Any]] = None

    def fit(self, X: NDArray[Any]) -> "TSNEAlgorithm":  # noqa: N803
        # TSNE doesn't have a separate fit method in most versions (it's fit_transform or nothing)
        # but we can store X to allow fit_transform later or just fit_transform now.
        self._X = X
        return self

    def transform(self, X: NDArray[Any]) -> NDArray[Any]:  # noqa: N803
        # TSNE does not support transform on new data.
        # It only supports fit_transform.
        return cast(NDArray[Any], self.model.fit_transform(X))

    def fit_transform(self, X: NDArray[Any]) -> NDArray[Any]:  # noqa: N803
        return cast(NDArray[Any], self.model.fit_transform(X))
