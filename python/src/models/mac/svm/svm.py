"""Support Vector Machine model suite."""

from typing import Any, cast

from sklearn.svm import SVC, SVR

from ..base import ClassicalModel


class SVMModel(ClassicalModel):
    """
    Support Vector Machine wrapper for classification (SVC) or regression (SVR).
    """

    def __init__(
        self, task: str = "regression", kernel: str = "rbf", **kwargs: Any
    ) -> None:
        """
        Initialize the SVM model.

        Args:
            task (str, optional): 'regression' or 'classification'. Defaults to "regression".
            kernel (str, optional): Kernel type (e.g., 'rbf', 'linear'). Defaults to "rbf".
            **kwargs: Additional arguments passed to the underlying sklearn model.
        """
        super().__init__()
        if task == "regression":
            self.model = SVR(kernel=cast(Any, kernel), **kwargs)
        else:
            self.model = SVC(kernel=cast(Any, kernel), **kwargs)


class SVRModel(SVMModel):
    """Support Vector Regression - Alias/Wrapper forcing regression task."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize SVRModel (Regression)."""
        super().__init__(task="regression", **kwargs)


class SVCModel(SVMModel):
    """Support Vector Classification - Alias/Wrapper forcing classification task."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize SVCModel (Classification)."""
        super().__init__(task="classification", **kwargs)
