"""
k-Nearest Neighbors models.
"""
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from .base import ClassicalModel

class kNNModel(ClassicalModel):
    def __init__(self, task='regression', n_neighbors=5, **kwargs):
        super().__init__()
        if task == 'regression':
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs)
        else:
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
