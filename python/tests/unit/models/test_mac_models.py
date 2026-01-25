import pytest
import torch

from python.src.models.time_series import TimeSeriesBackbone


class TestRegressionModels:
    @pytest.mark.parametrize(
        "model_name",
        [
            "LinearRegression",
            "Ridge",
            "Lasso",
            "ElasticNet",
            "LARS",
            "Polynomial",
            "OLSR",
            "Stepwise",
            "XGBoost",
            "LightGBM",
            "DecisionTree",
            "CART",
            "ID3",
            "C45",
            "C50",
            "CHAID",
            "DecisionStump",
            "ConditionalTree",
            "AdaBoost",
            "Bagging",
            "Stacking",
            "Voting",
            "WeightedAverage",
            "AdaBoost",
            "Bagging",
            "Stacking",
            "Voting",
            "WeightedAverage",
            "GBRT",
            "SVR",
            "LinearSVM",
            "NuSVM",
            "LSSVM",
        ],
    )
    def test_regression_integration(self, model_name, mac_dummy_input):
        cfg = {
            "name": model_name,
            "feature_dim": 10,
            "output_dim": 1,
            "model_kwargs": {},
        }
        backbone = TimeSeriesBackbone(cfg)
        with torch.no_grad():
            out = backbone(mac_dummy_input)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 1)


# Moved out of TestRegressionModels class as they no longer use 'self'
def test_olsr_model(regression_data):
    from python.src.models.mac import OLSRModel

    X, y = regression_data
    model = OLSRModel()
    model.fit(X, y)
    out = model(X)
    assert out.shape == (50, 1)
    assert out.dtype == torch.float32


def test_stepwise_model(regression_data):
    from python.src.models.mac import StepwiseRegressionModel

    X, y = regression_data
    model = StepwiseRegressionModel(n_features_to_select=2)
    model.fit(X, y)
    out = model(X)
    assert out.shape == (50, 1)
    # Check if only 2 features were selected in the internal model
    assert model.selected_features_ is not None
    assert model.selected_features_.sum() == 2


def test_mars_model(regression_data):
    from python.src.models.mac import MARSModel

    X, y = regression_data
    model = MARSModel(n_segments=3)
    model.fit(X, y)
    out = model(X)
    assert out.shape == (50, 1)


def test_loess_model(regression_data):
    from python.src.models.mac import LOESSModel

    X, y = regression_data
    # LOESS needs sorted X for stable interpolation test, but our model handles it
    model = LOESSModel(frac=0.5)
    model.fit(X, y)
    out = model(X)
    assert out.shape == (50, 1)


def test_lwl_model(regression_data):
    from python.src.models.mac import LWLModel

    model = LWLModel(n_neighbors=5)
    model.fit(regression_data[0], regression_data[1])
    out = model(regression_data[0])
    assert out.shape == (50, 1)
    assert out.dtype == torch.float32


def test_m5_model(regression_data):
    from python.src.models.mac import M5Model

    model = M5Model()
    model.fit(regression_data[0], regression_data[1])
    out = model(regression_data[0])
    assert out.shape == (50, 1)
    assert out.dtype == torch.float32


def test_lars_model(regression_data):
    from python.src.models.mac import LARSModel

    model = LARSModel()
    model.fit(regression_data[0], regression_data[1])
    out = model(regression_data[0])
    assert out.shape == (50, 1)


@pytest.mark.parametrize(
    "model_name",
    [
        "DecisionTree",
        "RandomForest",
        "GradientBoosting",
        "XGBoost",
        "LightGBM",
        "CART",
        "ID3",
        "C45",
        "C50",
        "CHAID",
        "DecisionStump",
        "ConditionalTree",
    ],
)
def test_tree_model_regression(model_name, regression_data):
    # Helper factory removed, using direct import logic for trees not in time_series.py?
    # Actually our test uses TimeSeriesBackbone or direct classes?
    # The test uses helper factory? Wait, I removed helper factory usage in my previous refactor.
    # Oh, the test_tree_boosting_integration uses HelperModelFactory BUT I removed it.
    # I need to fix this test to use direct imports or register them in helper factory if I want?
    # User asked to remove REGRESSION models from helper factory.
    # Tree models are often used for regression too.
    # But ID3/C45 are typically classification.
    # Let's instantiate directly based on name.

    if model_name == "RandomForest":
        from python.src.models.mac import RandomForestModel

        model = RandomForestModel(task="regression")
    elif model_name == "DecisionTree":
        from python.src.models.mac import DecisionTreeModel

        model = DecisionTreeModel(task="regression")
    elif model_name == "GradientBoosting":
        from python.src.models.mac import GradientBoostingModel

        model = GradientBoostingModel(task="regression")
    elif model_name == "XGBoost":
        from python.src.models.mac import XGBoostModel

        model = XGBoostModel(task="regression")
    elif model_name == "LightGBM":
        from python.src.models.mac import LightGBMModel

        model = LightGBMModel(task="regression")
    elif model_name == "CART":
        from python.src.models.mac import CARTModel

        model = CARTModel(task="regression")
    elif model_name == "ID3":
        from python.src.models.mac import ID3Model

        model = ID3Model(task="regression")  # Will fallback to default
    elif model_name == "C45":
        from python.src.models.mac import C45Model

        model = C45Model(task="regression")
    elif model_name == "C50":
        from python.src.models.mac import C50Model

        model = C50Model(task="regression")
    elif model_name == "CHAID":
        from python.src.models.mac import CHAIDModel

        model = CHAIDModel(task="regression")
    elif model_name == "DecisionStump":
        from python.src.models.mac import DecisionStumpModel

        model = DecisionStumpModel(task="regression")
    elif model_name == "ConditionalTree":
        from python.src.models.mac import ConditionalDecisionTreeModel

        model = ConditionalDecisionTreeModel(task="regression")
    else:
        pytest.skip(f"Model {model_name} not implemented in test harness")

    model.fit(regression_data[0], regression_data[1])
    out = model(regression_data[0])
    assert out.shape == (50, 1)


class TestClassificationModels:
    @pytest.mark.parametrize(
        "model_name",
        [
            "LogisticRegression",
            "NaiveBayes",
            "GaussianNB",
            "MultinomialNB",
            "AODE",
            "BayesianNetwork",
            "BBN",
            "LinearSVM",
            "NuSVM",
            "TWSVM",
            "OneClassSVM",
        ],
    )
    def test_classification_integration(self, model_name, mac_dummy_input):
        cfg = {
            "name": model_name,
            "feature_dim": 10,
            "output_dim": 1,
            "task": "classification",
            "type": "gaussian",  # for NaiveBayes (others ignore or use their own default)
            "model_kwargs": {},
        }

        # MultinomialNB fails with negative values (Gaussian random input contains negatives)
        # We need nonnegative input for MultinomialNB
        if "Multinomial" in model_name:
            mac_dummy_input = torch.abs(mac_dummy_input)

        backbone = TimeSeriesBackbone(cfg)
        with torch.no_grad():
            out = backbone(mac_dummy_input)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 1)


class TestTreeAndBoostingModels:
    @pytest.mark.parametrize(
        "model_name",
        [
            "DecisionTree",
            "RandomForest",
            "GradientBoosting",
            "XGBoost",
            "LightGBM",
            "CART",
            "ID3",
            "C45",
            "C50",
            "CHAID",
            "DecisionStump",
            "ConditionalTree",
            # M5 is special (regression only and custom), tested separately or added here?
            # Added separately below for detailed check.
        ],
    )
    def test_tree_boosting_integration(self, model_name, mac_dummy_input):
        # Determine task based on model nature?
        # Most can be regression. ID3/C45 default to classification in our implementation config,
        # but configured for regression in test?
        # TimeSeriesBackbone configures task based on cfg.
        # Let's force regression task since mac_dummy_input is float and we check regression shape.

        cfg = {
            "name": model_name,
            "feature_dim": 10,
            "output_dim": 1,
            "task": "regression",
        }

        # ID3/C45/C50 might default to classification in trees.py if not careful.
        # trees.py implementation respects task="regression".

        backbone = TimeSeriesBackbone(cfg)
        with torch.no_grad():
            out = backbone(mac_dummy_input)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 1)


class TestOtherModels:
    @pytest.mark.parametrize(
        "model_name",
        ["kNN", "SVM"],
    )
    def test_others_integration(self, model_name, mac_dummy_input):
        cfg = {"name": model_name, "feature_dim": 10, "output_dim": 1}
        backbone = TimeSeriesBackbone(cfg)
        with torch.no_grad():
            out = backbone(mac_dummy_input)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 1)


class TestClassicalCommon:
    def test_classical_model_fit(self, classical_cfg):
        backbone = TimeSeriesBackbone(classical_cfg)
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        from python.src.models.mac.base import ClassicalModel

        # Static analysis tools can sometimes misidentify nn.Module attributes as Tensors
        # Explicitly checking the type helps resolve "Object of type 'Tensor' is not callable"
        if isinstance(backbone.model, ClassicalModel):
            backbone.model.fit(X, y)
        assert backbone.model._is_fitted
        out = backbone(X)
        assert out.shape == (100, 1)
        assert torch.abs(out).sum() > 0

    def test_classical_model_sequence_output(self):
        cfg = {
            "name": "LinearRegression",
            "feature_dim": 10,
            "output_dim": 1,
            "return_sequence": True,
        }
        backbone = TimeSeriesBackbone(cfg)
        x = torch.randn(4, 30, 10)
        out = backbone(x)
        assert out.shape == (4, 30, 1)
