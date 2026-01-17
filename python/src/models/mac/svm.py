"""
Support Vector Machine models.
"""

from sklearn.svm import SVC, SVR

from .base import ClassicalModel


class SVMModel(ClassicalModel):
    def __init__(self, task="regression", kernel="rbf", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = SVR(kernel=kernel, **kwargs)
        else:
            self.model = SVC(kernel=kernel, **kwargs)
