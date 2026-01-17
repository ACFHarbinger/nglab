"""
Base class for classical machine learning models.
"""

from abc import ABC

import numpy as np
import torch
from torch import nn


class ClassicalModel(nn.Module, ABC):
    """
    Abstract base class for classical machine learning models.
    Wraps scikit-learn/XGBoost/LightGBM models to be used within NGLab.
    """

    def __init__(self, output_type="prediction"):
        super().__init__()
        self.output_type = output_type
        self.model = None  # To be initialized by subclasses
        self._is_fitted = False

        # Dummy parameter to ensure optimizer/device placement works if needed
        # though classical models typically run on CPU via sklearn.
        self.dummy_param = nn.Parameter(torch.empty(0))

    def _create_model(self, **kwargs):
        """Optional: Create the underlying classical model instance."""
        return None

    def forward(self, x, return_embedding=None, return_sequence=False):
        """
        Forward pass for inference.
        x: (Batch, Features) or (Batch, Seq, Features)
        """
        # Convert to numpy
        device = x.device
        x_np = x.detach().cpu().numpy()

        # Handle sequence data (Batch, Seq, Feat) -> (Batch * Seq, Feat) or use last step
        is_seq = x_np.ndim == 3
        if is_seq:
            b, s, f = x_np.shape
            # For classical models, we often just want the last step or flattened
            # If return_sequence is False, we take the last step.
            if not return_sequence:
                x_np = x_np[:, -1, :]
            else:
                x_np = x_np.reshape(b * s, f)

        # Classical models typically don't support batching in the same way,
        # but sklearn's predict handles [n_samples, n_features].
        if self.model is None or not self._is_fitted:
            # Return zeros if not fitted (safe fallback for initialization)
            batch_size = x_np.shape[0]
            out_np = np.zeros((batch_size, 1), dtype=np.float32)
        else:
            out_np = self.model.predict(x_np)
            if out_np.ndim == 1:
                out_np = out_np[:, np.newaxis]

        # Convert back to tensor
        out = torch.from_numpy(out_np).to(device).to(torch.float32)

        # Reshape back if sequence was requested
        if is_seq and return_sequence:
            out = out.view(b, s, -1)

        return out

    def fit(self, X, y):
        """Fit the underlying model."""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        if X.ndim == 3:
            # Flatten sequence for fitting classical models
            b, s, f = X.shape
            X = X.reshape(b * s, f)
            y = y.reshape(b * s, -1)

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        self.model.fit(X, y)
        self._is_fitted = True

    def state_dict(self, *args, **kwargs):
        """Override to include the classical model state if needed."""
        sd = super().state_dict(*args, **kwargs)
        if self.model is not None and self._is_fitted:
            sd["_classical_model"] = self.model
        return sd

    def load_state_dict(self, state_dict, strict=True):
        """Override to load the classical model state."""
        if "_classical_model" in state_dict:
            self.model = state_dict.pop("_classical_model")
            self._is_fitted = True
        return super().load_state_dict(state_dict, strict)
