"""Locally Weighted Learning model implementation."""

from typing import Any, Literal, cast

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ..base import ClassicalModel


class LWLModel(ClassicalModel):
    """
    Locally Weighted Learning (LWL).
    Implemented as k-Nearest Neighbors with distance-based weighting.
    """

    def __init__(
        self,
        task: str = "regression",
        n_neighbors: int = 5,
        kernel: str = "distance",
        **kwargs: Any,
    ) -> None:
        """Initialize LWLModel."""
        super().__init__()
        weights = cast(
            Literal["uniform", "distance"],
            kernel if kernel in ["distance", "uniform"] else "distance",
        )

        if task == "regression":
            self.model = KNeighborsRegressor(
                n_neighbors=n_neighbors, weights=weights, **kwargs
            )
        else:
            self.model = KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, **kwargs
            )
