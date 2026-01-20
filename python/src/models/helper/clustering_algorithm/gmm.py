from sklearn.mixture import GaussianMixture


class GMMAlgorithm:
    def __init__(self, n_components=1, **kwargs):
        self.model = GaussianMixture(n_components=n_components, **kwargs)

    def fit(self, X):  # noqa: N803
        self.model.fit(X)
        return self

    def predict(self, X):  # noqa: N803
        return self.model.predict(X)

    def fit_predict(self, X):  # noqa: N803
        return self.model.fit_predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
