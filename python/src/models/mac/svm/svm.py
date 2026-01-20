"""SVM Model."""

from sklearn.svm import SVC, SVR

from ..base import ClassicalModel


class SVMModel(ClassicalModel):
    """
    Support Vector Machine wrapper for classification (SVC) or regression (SVR).
    """

    def __init__(self, task="regression", kernel="rbf", **kwargs):
        """
        Initialize the SVM model.

        Args:
            task (str, optional): 'regression' or 'classification'. Defaults to "regression".
            kernel (str, optional): Kernel type (e.g., 'rbf', 'linear'). Defaults to "rbf".
            **kwargs: Additional arguments passed to the underlying sklearn model.
        """
        super().__init__()
        if task == "regression":
            self.model = SVR(kernel=kernel, **kwargs)
        else:
            self.model = SVC(kernel=kernel, **kwargs)


class SVRModel(SVMModel):
    """Support Vector Regression - Alias/Wrapper forcing regression task."""

    def __init__(self, **kwargs):
        super().__init__(task="regression", **kwargs)
