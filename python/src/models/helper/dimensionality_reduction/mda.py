
"""Mixture Discriminant Analysis (MDA) algorithm implementation."""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder


class MDAAlgorithm:
    """Mixture Discriminant Analysis (MDA) Algorithm."""

    def __init__(self, n_components_per_class: int = 1, **kwargs: Any) -> None:
        """Initialize MDA."""
        self.n_components_per_class = n_components_per_class
        self.gmms: dict[int, GaussianMixture] = {}
        self.classes_: NDArray[Any] | None = None
        self.priors_: NDArray[np.float64] | None = None
        self.le = LabelEncoder()

    def fit(self, x: NDArray[Any], y: NDArray[Any]) -> "MDAAlgorithm":
        """Fit the model."""
        y_encoded = cast(NDArray[np.int_], self.le.fit_transform(y))
        self.classes_ = cast(NDArray[Any], self.le.classes_)
        n_classes = len(cast(Any, self.classes_))
        self.gmms = {}
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        for c in range(n_classes):
            x_c = x[y_encoded == c]
            if x_c.shape[0] < self.n_components_per_class:
                # Fallback if not enough samples
                comp = 1
            else:
                comp = self.n_components_per_class

            gmm = GaussianMixture(n_components=comp, covariance_type="full")
            gmm.fit(x_c)
            self.gmms[c] = gmm
            self.priors_[c] = x_c.shape[0] / x.shape[0]

        return self

    def transform(self, x: NDArray[Any]) -> NDArray[np.float64]:
        """Apply dimensionality reduction/transformation."""
        if self.classes_ is None or self.priors_ is None:
            raise ValueError("Model not fitted yet.")

        n_samples = x.shape[0]
        n_classes = len(self.classes_)
        log_probs: NDArray[np.float64] = np.zeros(
            (n_samples, n_classes), dtype=np.float64
        )

        for c in range(n_classes):
            # weighted log prob
            log_probs[:, c] = self.gmms[c].score_samples(x) + np.log(
                self.priors_[c] + 1e-9
            )

        # Softmax: Exp-normalize
        max_log = np.max(log_probs, axis=1, keepdims=True)
        exp_log = np.exp(log_probs - max_log)
        probs = exp_log / np.sum(exp_log, axis=1, keepdims=True)

        return cast(NDArray[np.float64], probs)

    def fit_transform(self, x: NDArray[Any], y: NDArray[Any]) -> NDArray[np.float64]:
        """Fit and transform."""
        self.fit(x, y)
        return self.transform(x)
