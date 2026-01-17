from sklearn.cluster import AgglomerativeClustering

class HierarchicalClusteringAlgorithm:
    def __init__(self, n_clusters=2, **kwargs):
        self.model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)

    def fit(self, X):
        self.model.fit(X)
        return self

    def fit_predict(self, X):
        return self.model.fit_predict(X)

    def predict(self, X):
        # AgglomerativeClustering does not have a predict method for new data
        # but we can return fit_predict results if fit on the same X.
        if hasattr(self.model, "labels_"):
            return self.model.labels_
        return self.model.fit_predict(X)
