"""
Base class for classical machine learning models.
"""

from abc import ABC
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
from torch import nn


class ClassicalModel(nn.Module, ABC):
    """
    Abstract base class for classical machine learning models.
    Wraps scikit-learn/XGBoost/LightGBM models to be used within NGLab.
    """

    def __init__(self, output_type: str = "prediction") -> None:
        """
        Initialize the Classical Model base.

        Args:
            output_type (str, optional): The type of output expected from the model.
                Defaults to "prediction".
        """
        super().__init__()
        self.output_type = output_type
        self.model: Any = None  # To be initialized by subclasses
        self._is_fitted = False

        # Dummy parameter to ensure optimizer/device placement works if needed
        # though classical models typically run on CPU via sklearn.
        self.dummy_param = nn.Parameter(torch.empty(0))

    def _create_model(self, **kwargs: Any) -> Any:
        """Optional: Create the underlying classical model instance."""
        return None

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for inference.
        x: (Batch, Features) or (Batch, Seq, Features)
        """
        # Convert to numpy
        device = x.device
        x_np = x.detach().cpu().numpy()

        # Handle sequence data (Batch, Seq, Feat) -> (Batch * Seq, Feat) or use last step
        is_seq = x_np.ndim == 3
        b, s = 0, 0
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
            out_np = np.asarray(self.model.predict(x_np))
            if out_np.ndim == 1:
                out_np = out_np[:, np.newaxis]

        # Convert back to tensor
        out = torch.from_numpy(out_np).to(device).to(torch.float32)

        # Reshape back if sequence was requested
        if is_seq and return_sequence:
            out = out.view(b, s, -1)

        return out

    def fit(self, X: torch.Tensor, y: torch.Tensor | None = None) -> None:  # noqa: N803
        """Fit the underlying model."""
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y

        if X_np.ndim == 3:
            # Flatten sequence for fitting classical models
            b, s, f = X_np.shape
            X_np = X_np.reshape(b * s, f)
            if y_np is not None:
                y_np = y_np.reshape(b * s, -1)

        if self.model is None:
            raise RuntimeError(
                "Model has not been initialized. Ensure a subclass sets `self.model`."
            )

        if y_np is not None:
            if y_np.ndim == 2 and y_np.shape[1] == 1:
                y_np = y_np.ravel()
            self.model.fit(X_np, y_np)
        else:
            self.model.fit(X_np)
        self._is_fitted = True

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """Override to include the classical model state if needed."""
        sd = cast(dict[str, Any], super().state_dict(*args, **kwargs))
        if self.model is not None and self._is_fitted:
            sd["_classical_model"] = self.model
        return sd

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Override to load the classical model state."""
        state_dict_copy = dict(state_dict)
        if "_classical_model" in state_dict_copy:
            self.model = state_dict_copy.pop("_classical_model")
            self._is_fitted = True
        return super().load_state_dict(state_dict_copy, strict=strict, assign=assign)
