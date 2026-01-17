"""
Factory for Classical and Supplemental ML Models.
"""

from .helper.clustering import (
    KMeansModel,
    HierarchicalClusteringModel,
    DBSCANModel,
    DBSCANModel,
    GMMModel,
    KMediansModel,
    EMModel,
)

from .helper.dim_reduction import (
    PCAModel,
    PCAModel,
    TSNEModel,
    LDAModel,
    PCRModel,
    PLSRModel,
    MDSModel,
    SammonMappingModel,
    ProjectionPursuitModel,
    MDAModel,
    QDAModel,
    FDAModel,
    UMAPModel,
)
from .helper.association_rule import (
    AprioriModel,
    AprioriModel,
    FPGrowthModel,
    EclatModel,
)

class HelperModelFactory:
    """
    Factory class to create instances of supplemental ML models.
    Supports Clustering, Dimensionality Reduction, and Association Rule Learning.
    """
    
    _MODELS = {
        # Clustering
        "kmeans": KMeansModel,
        "hierarchical": HierarchicalClusteringModel,
        "dbscan": DBSCANModel,
        "dbscan": DBSCANModel,
        "gmm": GMMModel,
        "em": EMModel,
        "kmedians": KMediansModel,
        
        # Dimensionality Reduction
        "pca": PCAModel,
        "tsne": TSNEModel,
        "pca": PCAModel,
        "tsne": TSNEModel,
        "lda": LDAModel,
        "pcr": PCRModel,
        "plsr": PLSRModel,
        "mds": MDSModel,
        "sammon": SammonMappingModel,
        "pp": ProjectionPursuitModel,
        "mda": MDAModel,
        "qda": QDAModel,
        "fda": FDAModel,
        "umap": UMAPModel,
        
        # Association Rule Learning
        "apriori": AprioriModel,
        "apriori": AprioriModel,
        "fpgrowth": FPGrowthModel,
        "eclat": EclatModel,
    }

    @classmethod
    def create_model(cls, model_name: str, **kwargs):
        """
        Create a model instance based on the provided name.
        
        Args:
            model_name: Name of the algorithm (e.g., 'kmeans', 'pca', 'apriori').
            **kwargs: Hyperparameters for the model.
            
        Returns:
            An instance of ClassicalModel.
        """
        model_class = cls._MODELS.get(model_name.lower())
        if model_class is None:
            raise ValueError(f"Unknown model type: {model_name}. Available: {list(cls._MODELS.keys())}")
        
        return model_class(**kwargs)

    @classmethod
    def list_available_models(cls):
        """Returns a list of all available model names."""
        return list(cls._MODELS.keys())
