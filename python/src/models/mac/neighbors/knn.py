"""k-Nearest Neighbors model suite."""

from typing import Any

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ..base import ClassicalModel


class kNNModel(ClassicalModel):  # noqa: N801
    """
    k-Nearest Neighbors wrapper for classification or regression.
    """

    def __init__(
        self, task: str = "regression", n_neighbors: int = 5, **kwargs: Any
    ) -> None:
        """
        Initialize the k-NN model.

        Args:
            task (str, optional): 'regression' or 'classification'. Defaults to "regression".
            n_neighbors (int, optional): Number of neighbors. Defaults to 5.
            **kwargs: Additional arguments passed to the underlying sklearn model.
        """
        super().__init__()
        if task == "regression":
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs)
        else:
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
