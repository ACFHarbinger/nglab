"""
Clustering models for NGLab.
"""

from typing import Any

from ..mac.base import ClassicalModel
from .clustering_algorithm.dbscan import DBSCANAlgorithm
from .clustering_algorithm.gmm import GMMAlgorithm
from .clustering_algorithm.hierarchical import HierarchicalClusteringAlgorithm
from .clustering_algorithm.kmeans import KMeansAlgorithm
from .clustering_algorithm.kmedians import KMediansAlgorithm


class ClusteringModel(ClassicalModel):
    """Base class for clustering models."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(output_type="cluster")


class KMeansModel(ClusteringModel):
    def __init__(self, n_clusters: int = 8, **kwargs: Any) -> None:
        super().__init__()
        self.model = KMeansAlgorithm(n_clusters=n_clusters, **kwargs)


class HierarchicalClusteringModel(ClusteringModel):
    def __init__(self, n_clusters: int = 2, **kwargs: Any) -> None:
        super().__init__()
        self.model = HierarchicalClusteringAlgorithm(n_clusters=n_clusters, **kwargs)


class DBSCANModel(ClusteringModel):
    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs: Any) -> None:
        super().__init__()
        self.model = DBSCANAlgorithm(eps=eps, min_samples=min_samples, **kwargs)


class GMMModel(ClusteringModel):
    def __init__(self, n_components: int = 1, **kwargs: Any) -> None:
        super().__init__()
        self.model = GMMAlgorithm(n_components=n_components, **kwargs)


class EMModel(GMMModel):
    """Expectation Maximisation (EM) clustering. Alias for Gaussian Mixture Model."""

    pass


class KMediansModel(ClusteringModel):
    def __init__(self, n_clusters: int = 8, **kwargs: Any) -> None:
        super().__init__()
        self.model = KMediansAlgorithm(n_clusters=n_clusters, **kwargs)
