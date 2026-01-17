"""
k-Nearest Neighbors models.
"""

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .base import ClassicalModel


class kNNModel(ClassicalModel):
    def __init__(self, task="regression", n_neighbors=5, **kwargs):
        super().__init__()
        if task == "regression":
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs)
        else:
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)


class LWLModel(ClassicalModel):
    """
    Locally Weighted Learning (LWL).
    Implemented as k-Nearest Neighbors with distance-based weighting.
    """
    def __init__(self, task="regression", n_neighbors=5, kernel="distance", **kwargs):
        super().__init__()
        # 'distance' weight creates a locally weighted model
        weights = kernel if kernel in ["distance", "uniform"] else "distance"
        
        if task == "regression":
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, **kwargs)
        else:
            # For classification, LWL is essentially weighted kNN voting
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, **kwargs)
