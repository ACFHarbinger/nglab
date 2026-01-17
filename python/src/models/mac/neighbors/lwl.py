"""Locally Weighted Learning Model."""

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from ..base import ClassicalModel


class LWLModel(ClassicalModel):
    """
    Locally Weighted Learning (LWL).
    Implemented as k-Nearest Neighbors with distance-based weighting.
    """
    def __init__(self, task="regression", n_neighbors=5, kernel="distance", **kwargs):
        super().__init__()
        weights = kernel if kernel in ["distance", "uniform"] else "distance"
        
        if task == "regression":
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, **kwargs)
        else:
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, **kwargs)
