"""Least-Squares SVM (LSSVM) implementation."""

from typing import Any

from sklearn.kernel_ridge import KernelRidge

from ..base import ClassicalModel


class LSSVMModel(ClassicalModel):
    """
    Least-Squares SVM.
    Mathematically equivalent to Kernel Ridge Regression.
    """

    def __init__(self, alpha: float = 1.0, kernel: str = "rbf", **kwargs: Any) -> None:
        """Initialize LSSVMModel."""
        super().__init__()
        self.model = KernelRidge(alpha=alpha, kernel=kernel, **kwargs)
