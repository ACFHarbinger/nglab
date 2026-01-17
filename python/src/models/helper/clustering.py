"""
Clustering models for NGLab.
"""

from ..mac.base import ClassicalModel
from .logic_clustering.kmeans import KMeansAlgorithm
from .logic_clustering.hierarchical import HierarchicalClusteringAlgorithm
from .logic_clustering.dbscan import DBSCANAlgorithm
from .logic_clustering.gmm import GMMAlgorithm

class ClusteringModel(ClassicalModel):
    """Base class for clustering models."""
    def __init__(self, **kwargs):
        super().__init__(output_type="cluster")

class KMeansModel(ClusteringModel):
    def __init__(self, n_clusters=8, **kwargs):
        super().__init__()
        self.model = KMeansAlgorithm(n_clusters=n_clusters, **kwargs)

class HierarchicalClusteringModel(ClusteringModel):
    def __init__(self, n_clusters=2, **kwargs):
        super().__init__()
        self.model = HierarchicalClusteringAlgorithm(n_clusters=n_clusters, **kwargs)

class DBSCANModel(ClusteringModel):
    def __init__(self, eps=0.5, min_samples=5, **kwargs):
        super().__init__()
        self.model = DBSCANAlgorithm(eps=eps, min_samples=min_samples, **kwargs)

class GMMModel(ClusteringModel):
    def __init__(self, n_components=1, **kwargs):
        super().__init__()
        self.model = GMMAlgorithm(n_components=n_components, **kwargs)
