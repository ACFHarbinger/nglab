"""One-Class SVM for anomaly detection."""

from typing import Any

import numpy as np
import torch
from sklearn.svm import OneClassSVM

from ..base import ClassicalModel


class OneClassSVMModel(ClassicalModel):
    """
    One-Class SVM for Anomaly Detection.
    Output is usually -1 (outlier) or 1 (inlier).
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize OneClassSVMModel."""
        super().__init__()
        self.model = OneClassSVM(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass for anomaly detection (predicts 1 for inliers, -1 for outliers)."""
        if not self._is_fitted:
            return torch.zeros((x.size(0), 1), device=x.device, dtype=torch.float32)

        device = x.device
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            x_np = x_np[:, -1, :]

        out_np = self.model.predict(x_np)
        if out_np.ndim == 1:
            out_np = out_np[:, np.newaxis]

        return torch.from_numpy(out_np).to(device).to(torch.float32)
