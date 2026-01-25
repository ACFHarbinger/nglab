"""
Tests for Active Learning module - samplers and uncertainty estimation.
"""

import numpy as np
import pytest
import torch
from torch import nn

from python.src.pipeline.active_learning.sampler import (
    BaldSampler,
    BaseSampler,
    EntropySampler,
    RandomSampler,
    UncertaintySampler,
)
from python.src.pipeline.active_learning.uncertainty import (
    QuantileHead,
    mc_dropout_predict,
    pinball_loss,
)


class TestUncertaintySampler:
    """Tests for the UncertaintySampler class."""

    def test_selects_highest_uncertainty(self):
        sampler = UncertaintySampler(budget=3)
        # Higher score = more uncertain
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])

        indices = sampler.select(scores)

        assert len(indices) == 3
        assert isinstance(indices, np.ndarray)
        # Should select indices 1, 3, 2 (0.9, 0.8, 0.3)
        assert set(indices) == {1, 3, 2}

    def test_handles_2d_scores(self):
        sampler = UncertaintySampler(budget=2)
        scores = torch.tensor([[0.1], [0.9], [0.5]])

        indices = sampler.select(scores)

        assert len(indices) == 2
        assert 1 in indices  # Highest
        assert 2 in indices  # Second highest

    def test_budget_equals_pool_size(self):
        sampler = UncertaintySampler(budget=5)
        scores = torch.rand(5)

        indices = sampler.select(scores)

        assert len(indices) == 5

    def test_returns_numpy_array(self):
        sampler = UncertaintySampler(budget=2)
        scores = torch.rand(10)

        indices = sampler.select(scores)

        assert isinstance(indices, np.ndarray)


class TestEntropySampler:
    """Tests for the EntropySampler class."""

    def test_selects_highest_entropy(self):
        sampler = EntropySampler(budget=2)

        # Create probability distributions
        # Low entropy (certain): [0.99, 0.01]
        # High entropy (uncertain): [0.5, 0.5]
        probs = torch.tensor(
            [
                [0.99, 0.01],  # Low entropy
                [0.5, 0.5],  # Max entropy for binary
                [0.8, 0.2],  # Medium entropy
            ]
        )

        indices = sampler.select(probs)

        assert len(indices) == 2
        # Should select index 1 (highest entropy) and 2 (medium)
        assert 1 in indices

    def test_multiclass_entropy(self):
        sampler = EntropySampler(budget=1)

        # 4-class predictions
        probs = torch.tensor(
            [
                [0.97, 0.01, 0.01, 0.01],  # Low entropy
                [0.25, 0.25, 0.25, 0.25],  # Maximum entropy (uniform)
                [0.7, 0.1, 0.1, 0.1],  # Medium entropy
            ]
        )

        indices = sampler.select(probs)

        assert len(indices) == 1
        assert indices[0] == 1  # Uniform distribution has max entropy

    def test_handles_near_zero_probs(self):
        sampler = EntropySampler(budget=1)

        # Should not fail with very small probabilities
        probs = torch.tensor(
            [
                [1e-10, 1.0 - 1e-10],
                [0.5, 0.5],
            ]
        )

        indices = sampler.select(probs)

        assert len(indices) == 1


class TestBaldSampler:
    """Tests for the BaldSampler class."""

    def test_selects_highest_bald_score(self):
        sampler = BaldSampler(budget=1)

        # mc_preds: [num_samples, pool_size, num_classes]
        # High disagreement sample
        mc_preds = torch.tensor(
            [
                [[0.9, 0.1], [0.5, 0.5], [0.8, 0.2]],  # Sample 1
                [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]],  # Sample 2 (disagrees on first)
                [[0.5, 0.5], [0.5, 0.5], [0.8, 0.2]],  # Sample 3
            ]
        )

        indices = sampler.select(mc_preds)

        assert len(indices) == 1
        # First sample has highest disagreement (BALD score)
        assert indices[0] == 0

    def test_consistent_predictions_low_bald(self):
        sampler = BaldSampler(budget=2)

        # All MC samples agree - low BALD score
        mc_preds = torch.ones(5, 10, 3) * 0.33  # Uniform predictions

        indices = sampler.select(mc_preds)

        assert len(indices) == 2
        # All should have similar scores, selection may vary


class TestRandomSampler:
    """Tests for the RandomSampler class."""

    def test_selects_correct_number(self):
        sampler = RandomSampler(budget=5)
        scores = torch.rand(20)

        indices = sampler.select(scores)

        assert len(indices) == 5
        assert len(set(indices)) == 5  # All unique

    def test_indices_in_valid_range(self):
        sampler = RandomSampler(budget=10)
        pool_size = 50
        scores = torch.rand(pool_size)

        indices = sampler.select(scores)

        assert all(0 <= idx < pool_size for idx in indices)

    def test_different_runs_different_results(self):
        sampler = RandomSampler(budget=3)
        scores = torch.rand(100)

        results = [tuple(sorted(sampler.select(scores))) for _ in range(10)]

        # Should have some variation (very unlikely all 10 are the same)
        assert len(set(results)) > 1

    def test_ignores_score_values(self):
        sampler = RandomSampler(budget=2)
        # All zeros - should still work
        scores = torch.zeros(10)

        indices = sampler.select(scores)

        assert len(indices) == 2


class TestPinballLoss:
    """Tests for the pinball loss function."""

    def test_basic_pinball_loss(self):
        # pred: [batch, num_quantiles]
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[2.0]])
        quantiles = [0.1, 0.5, 0.9]

        loss = pinball_loss(pred, target, quantiles)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0

    def test_perfect_prediction_low_loss(self):
        # Prediction exactly matches target
        pred = torch.tensor([[5.0, 5.0, 5.0]])
        target = torch.tensor([[5.0]])
        quantiles = [0.1, 0.5, 0.9]

        loss = pinball_loss(pred, target, quantiles)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_asymmetric_loss(self):
        # Overestimate
        pred_over = torch.tensor([[3.0]])
        # Underestimate
        pred_under = torch.tensor([[1.0]])
        target = torch.tensor([[2.0]])
        quantiles = [0.9]  # 90th percentile

        loss_over = pinball_loss(pred_over, target, quantiles)
        loss_under = pinball_loss(pred_under, target, quantiles)

        # For 90th percentile, underestimating is penalized more
        assert loss_under > loss_over

    def test_batch_processing(self):
        batch_size = 16
        num_quantiles = 3

        pred = torch.randn(batch_size, num_quantiles)
        target = torch.randn(batch_size, 1)
        quantiles = [0.1, 0.5, 0.9]

        loss = pinball_loss(pred, target, quantiles)

        assert loss.ndim == 0
        assert not torch.isnan(loss)


class TestQuantileHead:
    """Tests for the QuantileHead module."""

    def test_output_shape(self):
        input_dim = 64
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        head = QuantileHead(input_dim, quantiles)

        x = torch.randn(32, input_dim)
        output = head(x)

        assert output.shape == (32, len(quantiles))

    def test_stores_quantiles(self):
        quantiles = [0.1, 0.5, 0.9]
        head = QuantileHead(128, quantiles)

        assert head.quantiles == quantiles

    def test_trainable_parameters(self):
        head = QuantileHead(64, [0.5])

        params = list(head.parameters())
        assert len(params) == 2  # weight and bias
        assert all(p.requires_grad for p in params)

    def test_integration_with_pinball_loss(self):
        head = QuantileHead(32, [0.1, 0.5, 0.9])
        optimizer = torch.optim.Adam(head.parameters())

        x = torch.randn(16, 32)
        target = torch.randn(16, 1)

        # One optimization step
        optimizer.zero_grad()
        pred = head(x)
        loss = pinball_loss(pred, target, head.quantiles)
        loss.backward()
        optimizer.step()

        # Loss should be valid
        assert not torch.isnan(loss)


class TestMCDropoutPredict:
    """Tests for MC Dropout prediction."""

    def test_basic_mc_dropout(self):
        class SimpleDropoutModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                return self.dropout(self.fc(x))

        model = SimpleDropoutModel()
        x = torch.randn(8, 10)

        result = mc_dropout_predict(model, x, n_samples=20)

        assert "mean" in result
        assert "variance" in result
        assert "std" in result
        assert result["mean"].shape == (8, 5)
        assert result["variance"].shape == (8, 5)

    def test_variance_with_dropout(self):
        class HighDropoutModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 3)
                self.dropout = nn.Dropout(0.8)  # High dropout

            def forward(self, x):
                return self.dropout(self.fc(x))

        model = HighDropoutModel()
        x = torch.randn(4, 5)

        result = mc_dropout_predict(model, x, n_samples=50)

        # With high dropout, there should be non-trivial variance
        # (unless we're very unlucky)
        assert result["variance"].sum() > 0

    def test_no_grad_context(self):
        class TrackingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 3)
                self.call_count = 0

            def forward(self, x):
                self.call_count += 1
                return self.fc(x)

        model = TrackingModel()
        x = torch.randn(2, 5, requires_grad=True)

        result = mc_dropout_predict(model, x, n_samples=5)

        assert model.call_count == 5
        # Results should not require grad
        assert not result["mean"].requires_grad
        assert not result["variance"].requires_grad

    def test_model_in_train_mode(self):
        class DropoutModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 3)
                self.dropout = nn.Dropout(0.5)
                self.train_mode_during_forward = None

            def forward(self, x):
                self.train_mode_during_forward = self.training
                return self.dropout(self.fc(x))

        model = DropoutModel()
        model.eval()  # Start in eval mode

        x = torch.randn(2, 5)
        mc_dropout_predict(model, x, n_samples=1)

        # Model should be in train mode during MC dropout
        assert model.train_mode_during_forward is True

    def test_std_is_sqrt_of_variance(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 3)
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                return self.dropout(self.fc(x))

        model = SimpleModel()
        x = torch.randn(4, 5)

        result = mc_dropout_predict(model, x, n_samples=20)

        expected_std = torch.sqrt(result["variance"])
        assert torch.allclose(result["std"], expected_std)


class TestBaseSampler:
    """Tests for the BaseSampler abstract class."""

    def test_abstract_select_raises(self):
        sampler = BaseSampler(budget=5)
        scores = torch.rand(10)

        with pytest.raises(NotImplementedError):
            sampler.select(scores)

    def test_stores_budget(self):
        sampler = BaseSampler(budget=42)
        assert sampler.budget == 42
