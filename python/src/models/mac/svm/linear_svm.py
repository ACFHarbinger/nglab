"""Linear SVM Model."""

from sklearn.svm import LinearSVC, LinearSVR

from ..base import ClassicalModel


class LinearSVMModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = LinearSVR(**kwargs)
        else:
            self.model = LinearSVC(**kwargs)
