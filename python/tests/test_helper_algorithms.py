import os
import sys

import torch

# Adjust path to include python/src
sys.path.append(os.path.join(os.getcwd(), "python", "src"))

from python.src.models.helper_factory import HelperModelFactory


def test_ml_factory_listing():
    models = HelperModelFactory.list_available_models()
    assert "kmeans" in models
    assert "pca" in models
    assert "apriori" in models
    assert len(models) == 21


def test_clustering_via_factory():
    X = torch.randn(20, 5)

    clustering_types = ["kmeans", "hierarchical", "dbscan", "gmm", "kmedians", "em"]
    for mtype in clustering_types:
        model = HelperModelFactory.create_model(mtype)
        model.fit(X, None)
        out = model(X)
        assert out.shape[0] == 20
        assert out.dtype == torch.float32


def test_dim_reduction_via_factory():
    X = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))

    # n_samples is 50, perplexity must be less than 50
    # n_samples is 50, perplexity must be less than 50
    dim_types = [
        "pca",
        "tsne",
        "lda",
        "pcr",
        "plsr",
        "mds",
        "sammon",
        "pp",
        "mda",
        "qda",
        "fda",
        "umap",
    ]
    for mtype in dim_types:
        if mtype == "tsne":
            model = HelperModelFactory.create_model(mtype, n_components=1, perplexity=5)
        elif mtype == "mda":
            # MDA needs enough samples per class
            model = HelperModelFactory.create_model(mtype, n_components_per_class=1)
        else:
            model = HelperModelFactory.create_model(mtype, n_components=1)

        # Models requiring y
        if mtype in ["lda", "plsr", "mda", "qda", "fda"]:
            model.fit(X, y)
        else:
            model.fit(X, None)

        out = model(X)
        assert out.shape[0] == 20
        assert out.shape[1] >= 1


def test_association_via_factory():
    X = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1]], dtype=torch.float32
    )

    assoc_types = ["apriori", "fpgrowth", "eclat"]
    for mtype in assoc_types:
        model = HelperModelFactory.create_model(mtype, min_support=0.1)
        model.fit(X, None)
        rules = model.get_rules()
        assert isinstance(rules, list)
        if rules:
            assert "antecedent" in rules[0]
            assert "confidence" in rules[0]
