from sklearn.cluster import DBSCAN

class DBSCANAlgorithm:
    def __init__(self, eps=0.5, min_samples=5, **kwargs):
        self.model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)

    def fit(self, X):
        self.model.fit(X)
        return self

    def fit_predict(self, X):
        return self.model.fit_predict(X)

    def predict(self, X):
        # DBSCAN does not have a predict method for new data.
        # It's an inductive model usually, but sklearn's is transductive.
        if hasattr(self.model, "labels_"):
            return self.model.labels_
        return self.model.fit_predict(X)
