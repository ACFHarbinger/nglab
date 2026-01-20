"""
Classical Machine Learning Model Factory.
"""

from typing import Any

from .mac import (
    AdaBoostModel,
    AODEModel,
    BaggingModel,
    BayesianNetworkModel,
    C45Model,
    C50Model,
    CARTModel,
    CHAIDModel,
    ConditionalDecisionTreeModel,
    DecisionStumpModel,
    DecisionTreeModel,
    ElasticNetModel,
    GaussianNaiveBayesModel,
    GBRTModel,
    GradientBoostingModel,
    LARSModel,
    LassoRegressionModel,
    LightGBMModel,
    LinearRegressionModel,
    LinearSVMModel,
    LOESSModel,
    LogisticRegressionModel,
    LSSVMModel,
    LWLModel,
    M5Model,
    MARSModel,
    MultinomialNaiveBayesModel,
    NaiveBayesModel,
    NuSVMModel,
    OLSRModel,
    OneClassSVMModel,
    PolynomialRegressionModel,
    RandomForestModel,
    RidgeRegressionModel,
    StackingModel,
    StepwiseRegressionModel,
    SVMModel,
    SVRModel,
    TWSVMModel,
    VotingModel,
    WeightedAverageModel,
    XGBoostModel,
    kNNModel,
)
from .mac.base import ClassicalModel

# List of MAC model names
MAC_MODEL_NAMES = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "LARS",
    "ElasticNet",
    "LogisticRegression",
    "Polynomial",
    "DecisionTree",
    "CART",
    "ID3",
    "C45",
    "C50",
    "CHAID",
    "DecisionStump",
    "ConditionalTree",
    "M5",
    "RandomForest",
    "GradientBoosting",
    "GBM",
    "GBRT",
    "AdaBoost",
    "Bagging",
    "Stacking",
    "Voting",
    "WeightedAverage",
    "Blending",
    "XGBoost",
    "LightGBM",
    "kNN",
    "SVM",
    "SVR",
    "LinearSVM",
    "NuSVM",
    "OneClassSVM",
    "LSSVM",
    "TWSVM",
    "NaiveBayes",
    "GaussianNB",
    "MultinomialNB",
    "AODE",
    "BayesianNetwork",
    "BBN",
    "BN",
    "OLSR",
    "Stepwise",
    "MARS",
    "LOESS",
    "LWL",
]


def create_mac_model(model_name: str, cfg: dict[str, Any]) -> ClassicalModel | None:  # noqa: PLR0911
    """
    Factory function to create classical machine learning models.

    Args:
        model_name: Name of the model to create.
        cfg: Configuration dictionary.

    Returns:
        Instantiated model or None if not a MAC model.
    """
    if model_name == "LinearRegression":
        return LinearRegressionModel(**cfg.get("model_kwargs", {}))
    elif model_name == "Ridge":
        return RidgeRegressionModel(
            alpha=cfg.get("alpha", 1.0), **cfg.get("model_kwargs", {})
        )
    elif model_name == "Lasso":
        return LassoRegressionModel(
            alpha=cfg.get("alpha", 1.0), **cfg.get("model_kwargs", {})
        )
    elif model_name == "LARS":
        return LARSModel(
            n_nonzero_coefs=cfg.get("n_nonzero_coefs", 500),
            **cfg.get("model_kwargs", {}),
        )
    elif model_name == "ElasticNet":
        return ElasticNetModel(
            alpha=cfg.get("alpha", 1.0),
            l1_ratio=cfg.get("l1_ratio", 0.5),
            **cfg.get("model_kwargs", {}),
        )
    elif model_name == "LogisticRegression":
        return LogisticRegressionModel(**cfg.get("model_kwargs", {}))
    elif model_name == "Polynomial":
        return PolynomialRegressionModel(
            degree=cfg.get("degree", 2), **cfg.get("model_kwargs", {})
        )
    elif model_name == "DecisionTree":
        return DecisionTreeModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "CART":
        return CARTModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "ID3":
        from .mac.trees import ID3Model

        return ID3Model(
            task=cfg.get("task", "classification"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "C45":
        return C45Model(
            task=cfg.get("task", "classification"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "C50":
        return C50Model(
            task=cfg.get("task", "classification"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "CHAID":
        return CHAIDModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "DecisionStump":
        return DecisionStumpModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "ConditionalTree":
        return ConditionalDecisionTreeModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "M5":
        return M5Model(**cfg.get("model_kwargs", {}))
    elif model_name == "RandomForest":
        return RandomForestModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name in {"GradientBoosting", "GBM"}:
        return GradientBoostingModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "GBRT":
        return GBRTModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "AdaBoost":
        return AdaBoostModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "Bagging":
        return BaggingModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "Stacking":
        return StackingModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "Voting":
        return VotingModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name in {"WeightedAverage", "Blending"}:
        return WeightedAverageModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "XGBoost":
        return XGBoostModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "LightGBM":
        return LightGBMModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "kNN":
        return kNNModel(
            task=cfg.get("task", "regression"),
            n_neighbors=cfg.get("n_neighbors", 5),
            **cfg.get("model_kwargs", {}),
        )
    elif model_name == "SVM":
        return SVMModel(
            task=cfg.get("task", "regression"),
            kernel=cfg.get("kernel", "rbf"),
            **cfg.get("model_kwargs", {}),
        )
    elif model_name == "SVR":
        return SVRModel(**cfg.get("model_kwargs", {}))
    elif model_name == "LinearSVM":
        return LinearSVMModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "NuSVM":
        return NuSVMModel(
            task=cfg.get("task", "regression"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "OneClassSVM":
        return OneClassSVMModel(**cfg.get("model_kwargs", {}))
    elif model_name == "LSSVM":
        return LSSVMModel(**cfg.get("model_kwargs", {}))
    elif model_name == "TWSVM":
        return TWSVMModel(**cfg.get("model_kwargs", {}))
    elif model_name == "NaiveBayes":
        return NaiveBayesModel(
            nb_type=cfg.get("type", "gaussian"), **cfg.get("model_kwargs", {})
        )
    elif model_name == "GaussianNB":
        return GaussianNaiveBayesModel(**cfg.get("model_kwargs", {}))
    elif model_name == "MultinomialNB":
        return MultinomialNaiveBayesModel(**cfg.get("model_kwargs", {}))
    elif model_name == "AODE":
        return AODEModel(**cfg.get("model_kwargs", {}))
    elif model_name in {"BayesianNetwork", "BBN", "BN"}:
        return BayesianNetworkModel(**cfg.get("model_kwargs", {}))
    elif model_name == "OLSR":
        return OLSRModel(**cfg.get("model_kwargs", {}))
    elif model_name == "Stepwise":
        return StepwiseRegressionModel(**cfg.get("model_kwargs", {}))
    elif model_name == "MARS":
        return MARSModel(**cfg.get("model_kwargs", {}))
    elif model_name == "LOESS":
        return LOESSModel(**cfg.get("model_kwargs", {}))
    elif model_name == "LWL":
        return LWLModel(
            task=cfg.get("task", "regression"),
            n_neighbors=cfg.get("n_neighbors", 5),
            **cfg.get("model_kwargs", {}),
        )

    return None
