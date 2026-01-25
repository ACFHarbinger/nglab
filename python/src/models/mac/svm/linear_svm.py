"""Linear SVM model implementation."""

from typing import Any

from sklearn.svm import LinearSVC, LinearSVR

from ..base import ClassicalModel


class LinearSVMModel(ClassicalModel):
    """Linear Support Vector Machine for classification or regression."""

    def __init__(self, task: str = "regression", **kwargs: Any) -> None:
        """Initialize LinearSVMModel."""
        super().__init__()
        if task == "regression":
            self.model = LinearSVR(**kwargs)
        else:
            self.model = LinearSVC(**kwargs)
