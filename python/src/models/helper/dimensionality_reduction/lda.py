from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDAAlgorithm:
    def __init__(self, n_components=None, **kwargs):
        self.model = LinearDiscriminantAnalysis(n_components=n_components, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def transform(self, X):
        return self.model.transform(X)

    def fit_transform(self, X, y):
        return self.model.fit_transform(X, y)

    def predict(self, X):
        return self.model.predict(X)
