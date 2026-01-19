"""
Dimensionality reduction models for NGLab.
"""

import torch
import numpy as np
from ..mac.base import ClassicalModel
from ..mac.linear import MARSModel
from .dimensionality_reduction.pca import PCAAlgorithm
from .dimensionality_reduction.tsne import TSNEAlgorithm
from .dimensionality_reduction.lda import LDAAlgorithm
from .dimensionality_reduction.sammon import SammonMappingAlgorithm
from .dimensionality_reduction.mda import MDAAlgorithm

# Sklearn imports for direct wrappers
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import MDS
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class DimReductionModel(ClassicalModel):
    """Base class for dimensionality reduction models."""

    def __init__(self, **kwargs):
        super().__init__(output_type="embedding")

    def forward(self, x, return_embedding=None, return_sequence=False):
        """Override forward to use transform instead of predict."""
        device = x.device
        x_np = x.detach().cpu().numpy()

        is_seq = x_np.ndim == 3
        if is_seq:
            b, s, f = x_np.shape
            if not return_sequence:
                x_np = x_np[:, -1, :]
            else:
                x_np = x_np.reshape(b * s, f)

        if self.model is None or not self._is_fitted:
            batch_size = x_np.shape[0]
            # Default to 2 components if unknown
            n_comp = getattr(self.model, "n_components", 2) or 2
            out_np = np.zeros((batch_size, n_comp), dtype=np.float32)
        else:
            # Most dim reduction models use transform
            if hasattr(self.model, "transform"):
                out_np = self.model.transform(x_np)
            else:
                out_np = self.model.fit_transform(x_np)

            if out_np.ndim == 1:
                out_np = out_np[:, np.newaxis]

        out = torch.from_numpy(out_np).to(device).to(torch.float32)

        if is_seq and return_sequence:
            out = out.view(b, s, -1)

        return out


class PCAModel(DimReductionModel):
    def __init__(self, n_components=None, **kwargs):
        super().__init__()
        self.model = PCAAlgorithm(n_components=n_components, **kwargs)


class TSNEModel(DimReductionModel):
    def __init__(self, n_components=2, **kwargs):
        super().__init__()
        self.model = TSNEAlgorithm(n_components=n_components, **kwargs)


class LDAModel(DimReductionModel):
    def __init__(self, n_components=None, **kwargs):
        super().__init__()
        self.model = LDAAlgorithm(n_components=n_components, **kwargs)

    def fit(self, X, y):
        """LDA needs y."""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        if X.ndim == 3:
            b, s, f = X.shape
            X = X.reshape(b * s, f)
            y = y.reshape(b * s, -1)

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        self.model.fit(X, y)
        self._is_fitted = True


class PCRModel(DimReductionModel):
    """Principal Component Regression (Dim Reduction aspect)."""

    def __init__(self, n_components=2, **kwargs):
        super().__init__()
        # Pipeline of PCA -> Regression, but model output is embedding (PCA part usually)
        # But PCR is a regression model.
        # User requested PCR as "dim reduction".
        # If used as DimReductionModel, we focus on the PCA transform, but PCR implies supervision?
        # Standard PCR uses PCA for dim reduction then Regresses.
        # Implemented here as a dim reduction that simply wraps PCA,
        # BUT since it's listed distinct from PCA, maybe they want the Regression capability?
        # ClassicalModel handles 'embedding' vs 'prediction'.
        # If output_type="embedding", returns PCA. If "prediction", returns regression.
        # I'll implement fit/predict logic.
        self.output_type = "embedding"  # Default
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.reg = LinearRegression()
        self.model = self.pca  # For transform

    def fit(self, X, y=None):
        if hasattr(X, "numpy"):
            X = X.cpu().numpy()
        self.pca.fit(X)
        X_pca = self.pca.transform(X)
        if y is not None:
            if hasattr(y, "numpy"):
                y = y.cpu().numpy()
            self.reg.fit(X_pca, y)
        self._is_fitted = True
        return self

    def forward(self, x, return_embedding=None, **kwargs):
        # Handle specialized PCR logic
        if self.output_type == "embedding" or return_embedding:
            return super().forward(x, return_embedding=True)
        # Prediction
        x_emb = super().forward(x, return_embedding=True).cpu().numpy()
        if x_emb.ndim == 3:
            x_emb = x_emb[:, -1, :]
        out_np = self.reg.predict(x_emb)
        if out_np.ndim == 1:
            out_np = out_np[:, np.newaxis]
        return torch.from_numpy(out_np).to(x.device).float()


class PLSRModel(DimReductionModel):
    def __init__(self, n_components=2, **kwargs):
        super().__init__()
        self.model = PLSRegression(n_components=n_components, **kwargs)

    def fit(self, X, y):
        # PLS needs y
        if hasattr(X, "numpy"):
            X = X.cpu().numpy()
        if hasattr(y, "numpy"):
            y = y.cpu().numpy()
        # Handle sequence flattening if needed (implied by base logic usually, but here explicit)
        if X.ndim == 3:
            X = X.reshape(X.shape[0] * X.shape[1], -1)
        if y is not None and y.ndim > 1:
            y = y.reshape(-1, y.shape[-1]) if y.ndim > 1 else y.ravel()
        elif y is not None:
            y = y.ravel()

        self.model.fit(X, y)
        self._is_fitted = True


class MDSModel(DimReductionModel):
    def __init__(self, n_components=2, **kwargs):
        super().__init__()
        self.model = MDS(n_components=n_components, **kwargs)


class SammonMappingModel(DimReductionModel):
    def __init__(self, n_components=2, **kwargs):
        super().__init__()
        self.model = SammonMappingAlgorithm(n_components=n_components, **kwargs)


class ProjectionPursuitModel(DimReductionModel):
    """Uses FastICA as proxy for Projection Pursuit."""

    def __init__(self, n_components=2, **kwargs):
        super().__init__()
        self.model = FastICA(n_components=n_components, **kwargs)


class QDAModel(DimReductionModel):
    def __init__(self, **kwargs):
        super().__init__()
        # Remove n_components if present as QDA does not use it
        if "n_components" in kwargs:
            kwargs.pop("n_components")
        self.model = QuadraticDiscriminantAnalysis(**kwargs)

    def fit(self, X, y):
        """QDA needs y."""
        # Reuse LDA fit logic essentially
        if hasattr(X, "numpy"):
            X = X.cpu().numpy()
        if hasattr(y, "numpy"):
            y = y.cpu().numpy()
        if X.ndim == 3:
            X = X.reshape(X.shape[0] * X.shape[1], -1)
        if y is not None:
            y = y.ravel()
        self.model.fit(X, y)
        self._is_fitted = True

    def forward(self, x, **kwargs):
        # QDA doesn't support transform usually, only predict/predict_proba
        # For embedding, can return predict_proba
        # Override to use predict_proba as 'embedding'
        if not self._is_fitted:
            return torch.zeros((x.shape[0], 1)).to(x.device)
        if hasattr(x, "numpy"):
            x = x.cpu().numpy()
        if x.ndim == 3:
            x = x[:, -1, :]
        out = self.model.predict_proba(x)
        return (
            torch.from_numpy(out)
            .to(x.device if hasattr(x, "device") else "cpu")
            .float()
        )


class MDAModel(DimReductionModel):
    def __init__(self, n_components_per_class=1, **kwargs):
        super().__init__()
        self.model = MDAAlgorithm(
            n_components_per_class=n_components_per_class, **kwargs
        )

    def fit(self, X, y):
        if hasattr(X, "numpy"):
            X = X.cpu().numpy()
        if hasattr(y, "numpy"):
            y = y.cpu().numpy()
        if X.ndim == 3:
            X = X.reshape(X.shape[0] * X.shape[1], -1)
        if y is not None:
            y = y.ravel()
        self.model.fit(X, y)
        self._is_fitted = True


class FDAModel(DimReductionModel):
    """
    Flexible Discriminant Analysis (FDA).
    Uses MARS (Multivariate Adaptive Regression Splines) to regress class labels,
    then performs LDA on the fitted values.
    """

    def __init__(self, n_components=None, **kwargs):
        super().__init__()
        self.n_components = n_components
        self.mars_kwargs = kwargs
        self.lda = LDAAlgorithm(n_components=n_components)
        self.mars_models = []
        self._is_fitted = False
        self.classes_ = None

    def fit(self, X, y):
        # 1. Prepare Data
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if X.ndim == 3:
            X = X.reshape(X.shape[0] * X.shape[1], -1)
        if y is not None:
            y = y.ravel()

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # 2. Optimal Scoring / Indicator Matrix Regression
        # Create dummy variables for separate regression
        # (Simplified FDA: One-vs-Rest regression for each class)
        self.mars_models = []

        # Fit X -> Class Indicator using MARS
        # Collect fitted values
        preds = []

        from sklearn.preprocessing import LabelBinarizer

        lb = LabelBinarizer()
        Y_dummies = lb.fit_transform(y)
        if Y_dummies.shape[1] == 1:  # Binary case returns single col
            Y_dummies = np.hstack([1 - Y_dummies, Y_dummies])

        for k in range(Y_dummies.shape[1]):
            # Train MARS for class k
            mars = MARSModel(**self.mars_kwargs)
            mars.fit(X, Y_dummies[:, k])
            self.mars_models.append(mars)

            # Predict (Fitted values)
            # MARSModel.predict returns (N, 1) usually or (N,)
            p = mars.predict(X)
            if p.ndim == 1:
                p = p[:, np.newaxis]
            preds.append(p)

        X_fitted = np.hstack(preds)

        # 3. Perform LDA on predicted/fitted values
        # LDA finds directions that discriminate the fitted class means
        self.lda.fit(X_fitted, y)
        self._is_fitted = True

    def transform(self, X):
        # Transform X -> MARS features -> LDA -> Low Dim
        preds = []
        for mars in self.mars_models:
            p = mars.predict(X)
            if p.ndim == 1:
                p = p[:, np.newaxis]
            preds.append(p)
        X_fitted = np.hstack(preds)
        return self.lda.transform(X_fitted)

    def forward(self, x, return_embedding=None, **kwargs):
        # Custom forward for FDA
        if not self._is_fitted:
            return torch.zeros((x.shape[0], self.n_components or 1)).to(x.device)

        device = x.device
        x_np = x.detach().cpu().numpy()
        is_seq = x_np.ndim == 3
        if is_seq:
            if not kwargs.get("return_sequence", False):
                x_np_flat = x_np[:, -1, :]
            else:
                b, s, f = x_np.shape
                x_np_flat = x_np.reshape(b * s, f)
        else:
            x_np_flat = x_np

        out_np = self.transform(x_np_flat)
        if out_np.ndim == 1:
            out_np = out_np[:, np.newaxis]

        out = torch.from_numpy(out_np).to(device).float()

        if is_seq and kwargs.get("return_sequence", False):
            out = out.view(b, s, -1)

        return out


class UMAPModel(DimReductionModel):
    def __init__(self, n_components=2, **kwargs):
        super().__init__()
        try:
            import umap

            self.model = umap.UMAP(n_components=n_components, **kwargs)
        except ImportError:
            # Fallback or error
            # For helper integration, better to have a dummy valid object or raise
            # Raising warning and fallback to PCA?
            print(
                "Warning: umap-learn not installed. Using PCA as fallback for UMAPModel."
            )
            self.model = PCA(n_components=n_components)
