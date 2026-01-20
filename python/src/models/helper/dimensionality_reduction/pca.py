from sklearn.decomposition import PCA


class PCAAlgorithm:
    def __init__(self, n_components=None, **kwargs):
        self.model = PCA(n_components=n_components, **kwargs)

    def fit(self, X):  # noqa: N803
        self.model.fit(X)
        return self

    def transform(self, X):  # noqa: N803
        return self.model.transform(X)

    def fit_transform(self, X):  # noqa: N803
        return self.model.fit_transform(X)

    def inverse_transform(self, X):  # noqa: N803
        return self.model.inverse_transform(X)
