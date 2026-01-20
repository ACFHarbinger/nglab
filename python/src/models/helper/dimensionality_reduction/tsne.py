from sklearn.manifold import TSNE


class TSNEAlgorithm:
    def __init__(self, n_components=2, **kwargs):
        self.model = TSNE(n_components=n_components, **kwargs)

    def fit(self, X):  # noqa: N803
        # TSNE doesn't have a separate fit method in most versions (it's fit_transform or nothing)
        # but we can store X to allow fit_transform later or just fit_transform now.
        self._X = X
        return self

    def transform(self, X):  # noqa: N803
        # TSNE does not support transform on new data.
        # It only supports fit_transform.
        return self.model.fit_transform(X)

    def fit_transform(self, X):  # noqa: N803
        return self.model.fit_transform(X)
