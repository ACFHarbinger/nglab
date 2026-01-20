"""Nu SVM Model."""

from sklearn.svm import NuSVC, NuSVR

from ..base import ClassicalModel


class NuSVMModel(ClassicalModel):
    def __init__(self, task="regression", nu=0.5, **kwargs):
        super().__init__()
        if task == "regression":
            self.model = NuSVR(nu=nu, **kwargs)
        else:
            self.model = NuSVC(nu=nu, **kwargs)
