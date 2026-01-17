"""
Classical linear models.
"""
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from .base import ClassicalModel

class LinearRegressionModel(ClassicalModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LinearRegression(**kwargs)

class RidgeRegressionModel(ClassicalModel):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__()
        self.model = Ridge(alpha=alpha, **kwargs)

class LassoRegressionModel(ClassicalModel):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__()
        self.model = Lasso(alpha=alpha, **kwargs)

class ElasticNetModel(ClassicalModel):
    def __init__(self, alpha=1.0, l1_ratio=0.5, **kwargs):
        super().__init__()
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs)

class LogisticRegressionModel(ClassicalModel):
    def __init__(self, **kwargs):
        super().__init__()
        # Ensure we have a sensible default for multi-class if needed
        self.model = LogisticRegression(max_iter=1000, **kwargs)

class PolynomialRegressionModel(ClassicalModel):
    def __init__(self, degree=2, **kwargs):
        super().__init__()
        self.model = Pipeline([
            ("poly_features", PolynomialFeatures(degree=degree)),
            ("linear_regression", LinearRegression(**kwargs))
        ])
        self._is_fitted = False
    
    def fit(self, X, y):
        # We need to override fit for pipeline if we want to be explicit, 
        # but sklearn pipeline.fit works fine.
        super().fit(X, y)
