from typing import Any, Optional, Dict
import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder


class MDAAlgorithm:
    """Mixture Discriminant Analysis (MDA) Algorithm."""
    
    def __init__(self, n_components_per_class: int = 1, **kwargs: Any) -> None:
        """Initialize MDA."""
        self.n_components_per_class = n_components_per_class
        self.gmms: Dict[int, GaussianMixture] = {}
        self.classes_: Optional[NDArray[Any]] = None
        self.priors_: Optional[NDArray[np.float64]] = None
        self.le = LabelEncoder()

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> "MDAAlgorithm":  # noqa: N803
        """Fit the model."""
        y_encoded = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        n_classes = len(self.classes_)
        self.gmms = {}
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        for c in range(n_classes):
            X_c = X[y_encoded == c]
            if X_c.shape[0] < self.n_components_per_class:
                # Fallback if not enough samples
                comp = 1
            else:
                comp = self.n_components_per_class

            gmm = GaussianMixture(n_components=comp, covariance_type="full")
            gmm.fit(X_c)
            self.gmms[c] = gmm
            self.priors_[c] = X_c.shape[0] / X.shape[0]

        return self

    def transform(self, X: NDArray[Any]) -> NDArray[np.float64]:  # noqa: N803
        """Apply dimensionality reduction/transformation."""
        if self.classes_ is None or self.priors_ is None:
            raise ValueError("Model not fitted yet.")
            
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes))

        for c in range(n_classes):
            # weighted log prob
            log_probs[:, c] = self.gmms[c].score_samples(X) + np.log(
                self.priors_[c] + 1e-9
            )

        # Softmax: Exp-normalize
        max_log = np.max(log_probs, axis=1, keepdims=True)
        exp_log = np.exp(log_probs - max_log)
        probs = exp_log / np.sum(exp_log, axis=1, keepdims=True)

        return probs

    def fit_transform(self, X: NDArray[Any], y: NDArray[Any]) -> NDArray[np.float64]:  # noqa: N803
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)
