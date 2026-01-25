"""Nu-SVM model implementation."""

from typing import Any

from sklearn.svm import NuSVC, NuSVR

from ..base import ClassicalModel


class NuSVMModel(ClassicalModel):
    """Nu-Support Vector Machine for classification or regression."""

    def __init__(
        self, task: str = "regression", nu: float = 0.5, **kwargs: Any
    ) -> None:
        """Initialize NuSVMModel."""
        super().__init__()
        if task == "regression":
            self.model = NuSVR(nu=nu, **kwargs)
        else:
            self.model = NuSVC(nu=nu, **kwargs)
