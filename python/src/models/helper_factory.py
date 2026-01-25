"""
Factory for Classical and Supplemental ML Models.
"""

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .mac.base import ClassicalModel

from .helper.association_rule import (
    AprioriModel,
    EclatModel,
    FPGrowthModel,
)
from .helper.clustering import (
    DBSCANModel,
    EMModel,
    GMMModel,
    HierarchicalClusteringModel,
    KMeansModel,
    KMediansModel,
)
from .helper.dim_reduction import (
    FDAModel,
    LDAModel,
    MDAModel,
    MDSModel,
    PCAModel,
    PCRModel,
    PLSRModel,
    ProjectionPursuitModel,
    QDAModel,
    SammonMappingModel,
    TSNEModel,
    UMAPModel,
)


class HelperModelFactory:
    """
    Factory class to create instances of supplemental ML models.
    Supports Clustering, Dimensionality Reduction, and Association Rule Learning.
    """

    _MODELS: ClassVar[dict[str, type]] = {
        # Clustering
        "kmeans": KMeansModel,
        "hierarchical": HierarchicalClusteringModel,
        "dbscan": DBSCANModel,
        "gmm": GMMModel,
        "em": EMModel,
        "kmedians": KMediansModel,
        # Dimensionality Reduction
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
        "fpgrowth": FPGrowthModel,
        "eclat": EclatModel,
    }

    @classmethod
    def create_model(cls, model_name: str, **kwargs: Any) -> "ClassicalModel":
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
            raise ValueError(
                f"Unknown model type: {model_name}. Available: {list(cls._MODELS.keys())}"
            )

        from typing import cast

        return cast("ClassicalModel", model_class(**kwargs))

    @classmethod
    def list_available_models(cls) -> list[str]:
        """Returns a list of all available model names."""
        return list(cls._MODELS.keys())
