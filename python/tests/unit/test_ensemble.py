from unittest.mock import patch

import pytest
import torch
from torch import nn

from python.src.models.ensemble import EnsembleModel, create_ensemble_from_configs


class SimpleModel(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val
        self.linear = nn.Linear(10, 5)  # Added for parameter registration if needed

    def forward(self, x):
        return torch.ones(x.shape[0], 5) * self.val


def test_ensemble_init_average():
    models = [SimpleModel(1.0), SimpleModel(2.0)]
    ensemble = EnsembleModel(models, strategy="average")
    assert ensemble.strategy == "average"
    assert len(ensemble.models) == 2
    assert torch.allclose(ensemble.weights, torch.tensor([0.5, 0.5]))


def test_ensemble_init_weighted():
    models = [SimpleModel(1.0), SimpleModel(2.0)]
    ensemble = EnsembleModel(models, strategy="weighted", weights=[0.3, 0.7])
    assert ensemble.strategy == "weighted"
    assert torch.allclose(ensemble.weights, torch.tensor([0.3, 0.7]))


def test_ensemble_forward_average():
    models = [SimpleModel(1.0), SimpleModel(3.0)]
    ensemble = EnsembleModel(models, strategy="average")
    x = torch.randn(2, 10)
    out = ensemble(x)
    assert out.shape == (2, 5)
    assert torch.allclose(out, torch.ones(2, 5) * 2.0)


def test_ensemble_forward_weighted():
    models = [SimpleModel(1.0), SimpleModel(3.0)]
    ensemble = EnsembleModel(models, strategy="weighted", weights=[0.25, 0.75])
    x = torch.randn(2, 10)
    out = ensemble(x)
    # 0.25 * 1.0 + 0.75 * 3.0 = 0.25 + 2.25 = 2.5
    assert torch.allclose(out, torch.ones(2, 5) * 2.5)


def test_ensemble_forward_voting():
    class ClassificationModel(nn.Module):
        def __init__(self, pred):
            super().__init__()
            self.pred = pred

        def forward(self, x):
            # Return logits where 'pred' index is highest
            out = torch.zeros(x.shape[0], 3)
            out[:, self.pred] = 1.0
            return out

    models = [ClassificationModel(0), ClassificationModel(1), ClassificationModel(1)]
    ensemble = EnsembleModel(models, strategy="voting")
    x = torch.randn(2, 10)
    out = ensemble(x)
    assert out.shape == (2,)
    assert torch.all(out == 1)  # Mode of [0, 1, 1] is 1


def test_ensemble_forward_stacking():
    models = [SimpleModel(1.0), SimpleModel(2.0)]
    meta = nn.Linear(10, 5)  # 2 models * 5 output_dim = 10
    ensemble = EnsembleModel(models, strategy="stacking", meta_learner=meta)
    x = torch.randn(2, 10)
    out = ensemble(x)
    assert out.shape == (2, 5)


def test_ensemble_forward_stacking_error():
    models = [SimpleModel(1.0)]
    ensemble = EnsembleModel(models, strategy="stacking")
    with pytest.raises(ValueError, match="requires a meta_learner"):
        ensemble(torch.randn(1, 10))


def test_ensemble_invalid_strategy():
    models = [SimpleModel(1.0)]
    ensemble = EnsembleModel(models, strategy="invalid")
    with pytest.raises(ValueError, match="Unknown strategy"):
        ensemble(torch.randn(1, 10))


def test_predict_with_uncertainty():
    models = [SimpleModel(1.0), SimpleModel(3.0)]
    ensemble = EnsembleModel(models)
    x = torch.randn(2, 10)
    res = ensemble.predict_with_uncertainty(x)
    assert "mean" in res
    assert "std" in res
    assert "predictions" in res
    assert torch.allclose(res["mean"], torch.ones(2, 5) * 2.0)
    # std of [1, 3] is 1.414... (sqrt(2))
    assert torch.allclose(
        res["std"], torch.ones(2, 5) * torch.tensor([1, 3], dtype=torch.float32).std()
    )


@patch("python.src.models.ensemble.TimeSeriesBackbone")
def test_create_ensemble_from_configs(mock_backbone):
    mock_backbone.return_value = SimpleModel(1.0)
    configs = [{"name": "M1"}, {"name": "M2"}]
    ensemble = create_ensemble_from_configs(
        configs, strategy="weighted", weights=[0.4, 0.6]
    )
    assert isinstance(ensemble, EnsembleModel)
    assert ensemble.n_models == 2
    assert ensemble.strategy == "weighted"
    assert mock_backbone.call_count == 2
