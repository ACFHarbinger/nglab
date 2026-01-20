import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder


class MDAAlgorithm:
    def __init__(self, n_components_per_class=1, **kwargs):
        self.n_components_per_class = n_components_per_class
        self.gmms = {}
        self.classes_ = None
        self.priors_ = None
        self.le = LabelEncoder()

    def fit(self, X, y):  # noqa: N803
        y = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        n_classes = len(self.classes_)
        self.gmms = {}
        self.priors_ = np.zeros(n_classes)

        for c in range(n_classes):
            X_c = X[y == c]
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

    def transform(self, X):  # noqa: N803
        # Return posterior probs per class as new features
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes))

        for c in range(n_classes):
            # weighted log prob
            log_probs[:, c] = self.gmms[c].score_samples(X) + np.log(
                self.priors_[c] + 1e-9
            )

        # Softmax
        # Exp-normalize
        max_log = np.max(log_probs, axis=1, keepdims=True)
        exp_log = np.exp(log_probs - max_log)
        probs = exp_log / np.sum(exp_log, axis=1, keepdims=True)

        return probs

    def fit_transform(self, X, y):  # noqa: N803
        self.fit(X, y)
        return self.transform(X)
