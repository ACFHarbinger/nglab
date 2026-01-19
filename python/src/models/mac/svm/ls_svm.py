"""LS-SVM Model."""

from sklearn.kernel_ridge import KernelRidge
from ..base import ClassicalModel


class LSSVMModel(ClassicalModel):
    """
    Least-Squares SVM.
    Mathematically equivalent to Kernel Ridge Regression.
    """

    def __init__(self, alpha=1.0, kernel="rbf", **kwargs):
        super().__init__()
        self.model = KernelRidge(alpha=alpha, kernel=kernel, **kwargs)
