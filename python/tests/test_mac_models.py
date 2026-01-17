import pytest
import torch

from python.src.models.time_series import TimeSeriesBackbone


@pytest.mark.parametrize(
    "model_name",
    [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "LogisticRegression",
        "Polynomial",
        "DecisionTree",
        "RandomForest",
        "GradientBoosting",
        "XGBoost",
        "LightGBM",
        "kNN",
        "SVM",
        "NaiveBayes",
    ],
)
def test_classical_models_integration(model_name, mac_dummy_input):
    """Test that classical models can be instantiated and produce output of correct shape."""
    cfg = {"name": model_name, "feature_dim": 10, "output_dim": 1, "model_kwargs": {}}

    # Optional: adjust task for classifier if needed
    if model_name in ["LogisticRegression", "NaiveBayes"]:
        cfg["task"] = "classification"
        cfg["type"] = "gaussian"

    backbone = TimeSeriesBackbone(cfg)

    # Try forward pass (should return zeros or default as not fitted)
    with torch.no_grad():
        out = backbone(mac_dummy_input)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 1)


def test_classical_model_fit(classical_cfg):
    """Test standard fitting process for a classical model."""
    backbone = TimeSeriesBackbone(classical_cfg)

    # Data: 100 samples, 10 features
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)

    # Fit
    backbone.model.fit(X, y)
    assert backbone.model._is_fitted

    # Predict
    out = backbone(X)
    assert out.shape == (100, 1)
    # Should not be all zeros anymore
    assert torch.abs(out).sum() > 0


def test_classical_model_sequence_output():
    """Test if classical models handle return_sequence=True."""
    cfg = {
        "name": "LinearRegression",
        "feature_dim": 10,
        "output_dim": 1,
        "return_sequence": True,
    }
    backbone = TimeSeriesBackbone(cfg)

    x = torch.randn(4, 30, 10)
    out = backbone(x)

    # Should be (Batch, Seq, OutputDim) -> (4, 30, 1)
    assert out.shape == (4, 30, 1)
