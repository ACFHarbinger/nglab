"""
Dimensionality reduction models for NGLab.
"""

import torch
import numpy as np
from ..mac.base import ClassicalModel
from .logic_dim_reduction.pca import PCAAlgorithm
from .logic_dim_reduction.tsne import TSNEAlgorithm
from .logic_dim_reduction.lda import LDAAlgorithm

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
            n_comp = getattr(self.model, 'n_components', 2) or 2
            out_np = np.zeros((batch_size, n_comp), dtype=np.float32)
        else:
            # Most dim reduction models use transform
            if hasattr(self.model, 'transform'):
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
