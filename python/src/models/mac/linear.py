"""
Classical linear models.
"""

from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    Lars,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np
import torch
from .base import ClassicalModel

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    lowess = None


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


class LARSModel(ClassicalModel):
    def __init__(self, n_nonzero_coefs=500, **kwargs):
        super().__init__()
        self.model = Lars(n_nonzero_coefs=n_nonzero_coefs, **kwargs)


class LogisticRegressionModel(ClassicalModel):
    def __init__(self, **kwargs):
        super().__init__()
        # Ensure we have a sensible default for multi-class if needed
        self.model = LogisticRegression(max_iter=1000, **kwargs)


class PolynomialRegressionModel(ClassicalModel):
    def __init__(self, degree=2, **kwargs):
        super().__init__()
        self.model = Pipeline(
            [
                ("poly_features", PolynomialFeatures(degree=degree)),
                ("linear_regression", LinearRegression(**kwargs)),
            ]
        )
        self._is_fitted = False

    def fit(self, X, y):
        # We need to override fit for pipeline if we want to be explicit,
        # but sklearn pipeline.fit works fine.
        super().fit(X, y)


class OLSRModel(LinearRegressionModel):
    """Ordinary Least Squares Regression - Alias for LinearRegressionModel."""
    pass


class StepwiseRegressionModel(ClassicalModel):
    """Stepwise Regression using Sequential Feature Selection."""
    def __init__(self, direction="forward", n_features_to_select="auto", **kwargs):
        super().__init__()
        self.base_estimator = LinearRegression()
        self.model = SequentialFeatureSelector(
            self.base_estimator,
            n_features_to_select=n_features_to_select,
            direction=direction,
            **kwargs
        )

    def fit(self, X, y):
        super().fit(X, y)
        # After selection, fit the base estimator on selected features
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        if X.ndim == 3:
            X = X.reshape(X.shape[0] * X.shape[1], -1)
            y = y.reshape(y.shape[0] * y.shape[1], -1)
            
        self.selected_features_ = self.model.get_support()
        self.final_model = LinearRegression()
        self.final_model.fit(X[:, self.selected_features_], y)
        self._is_fitted = True

    def forward(self, x, **kwargs):
        # Override forward to use final_model
        if not self._is_fitted:
            return super().forward(x, **kwargs)
        
        device = x.device
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            b, s, f = x_np.shape
            x_np = x_np[:, -1, :] # Default to last step
            
        out_np = self.final_model.predict(x_np[:, self.selected_features_])
        if out_np.ndim == 1:
            out_np = out_np[:, np.newaxis]
            
        return torch.from_numpy(out_np).to(device).to(torch.float32)

    def predict(self, X):
        if not self._is_fitted:
            return np.zeros((X.shape[0], 1))
        return self.final_model.predict(X[:, self.selected_features_])


class MARSModel(ClassicalModel):
    """Simplified Multivariate Adaptive Regression Splines (Piecewise Linear)."""
    def __init__(self, n_segments=5, **kwargs):
        super().__init__()
        self.n_segments = n_segments
        # In a real MARS, basis functions are added greedily. 
        # Here we use a simplified piecewise linear approach via bins.
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree=1)), # Placeholder for complex basis
            ("linear", LinearRegression(**kwargs))
        ])

    def fit(self, X, y):
        # Implementation of simplified basis functions: max(0, x - t) and max(0, t - x)
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        # simplified MARS: generate hinge functions for each feature
        hinges = []
        for i in range(X.shape[1]):
            feat = X[:, i]
            knots = np.linspace(feat.min(), feat.max(), self.n_segments)
            for knot in knots:
                hinges.append(np.maximum(0, feat - knot))
                hinges.append(np.maximum(0, knot - feat))
        
        X_hinge = np.column_stack(hinges)
        self.model = LinearRegression() # Use self.model for base class compatibility
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        if y is not None and y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
            
        self.model.fit(X_hinge, y)
        self.knots_ = knots # Simplified: assumes same intervals for all feats
        self._is_fitted = True
        
    def forward(self, x, **kwargs):
        # Override forward to handle hinge transformation
        if not self._is_fitted:
            return super().forward(x, **kwargs)
            
        device = x.device
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            x_np = x_np[:, -1, :]
            
        hinges = []
        for i in range(x_np.shape[1]):
            feat = x_np[:, i]
            knots = np.linspace(feat.min(), feat.max(), self.n_segments)
            for knot in knots:
                hinges.append(np.maximum(0, feat - knot))
                hinges.append(np.maximum(0, knot - feat))
        
        X_hinge = np.column_stack(hinges)
        out_np = self.model.predict(X_hinge)
        if out_np.ndim == 1:
            out_np = out_np[:, np.newaxis]
            
        return torch.from_numpy(out_np).to(device).to(torch.float32)

    def predict(self, X):
        if not self._is_fitted:
            return np.zeros((X.shape[0], 1))
            
        hinges = []
        for i in range(X.shape[1]):
            feat = X[:, i]
            # knots must be the same as in fit
            knots = np.linspace(feat.min(), feat.max(), self.n_segments)
            for knot in knots:
                hinges.append(np.maximum(0, feat - knot))
                hinges.append(np.maximum(0, knot - feat))
        
        X_hinge = np.column_stack(hinges)
        return self.model.predict(X_hinge)


class LOESSModel(ClassicalModel):
    """Locally Estimated Scatterplot Smoothing (LOESS)."""
    def __init__(self, frac=0.66, it=3, **kwargs):
        super().__init__()
        self.frac = frac
        self.it = it
        self.kwargs = kwargs

    def fit(self, X, y):
        if lowess is None:
            raise ImportError("statsmodels is required for LOESSModel")
        
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
            
        if X.ndim == 3:
            X = X.reshape(X.shape[0] * X.shape[1], -1)
            y = y.reshape(y.shape[0] * y.shape[1], -1)

        # LOESS is typically univariate for the 'x' axis (time or index)
        # If X is multivariate, we flatten or use the first principal component/index
        self.X_train = X
        self.y_train = y
        self._is_fitted = True

    def predict(self, X):
        if not self._is_fitted:
            return np.zeros((X.shape[0], 1))
            
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = X

        # LOESS usually works by smoothing existing data. 
        # For new X, we interpolate or re-run lowess on combined data.
        # Simplified: lowess on y_train using first feature of X_train
        # This is a basic approximation for 'smoothing'.
        
        # We'll use the first column of X as the independent variable for smoothing
        x_axis = self.X_train[:, 0]
        y_axis = self.y_train.ravel()
        
        # Sort by x_axis for lowess
        indices = np.argsort(x_axis)
        
        if lowess is None:
            raise ImportError("statsmodels is required for LOESSModel")
             
        res = lowess(y_axis[indices], x_axis[indices], frac=self.frac, it=self.it, **self.kwargs)
        
        # res returns [sorted_x, smoothed_y]
        # We interpolate to get values for requested X
        from scipy.interpolate import interp1d
        f = interp1d(res[:, 0], res[:, 1], bounds_error=False, fill_value="extrapolate") # pyright: ignore[reportArgumentType]
        
        out = f(X_np[:, 0])
        if out.ndim == 1:
            out = out[:, np.newaxis]
        return out

    def forward(self, x, **kwargs):
        # Override forward to use numpy predict
        device = x.device
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            x_np = x_np[:, -1, :]
            
        out_np = self.predict(x_np)
        return torch.from_numpy(out_np).to(device).to(torch.float32)
