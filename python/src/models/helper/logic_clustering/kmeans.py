from sklearn.cluster import KMeans

class KMeansAlgorithm:
    def __init__(self, n_clusters=8, **kwargs):
        self.model = KMeans(n_clusters=n_clusters, **kwargs)

    def fit(self, X):
        self.model.fit(X)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fit_predict(self, X):
        return self.model.fit_predict(X)
