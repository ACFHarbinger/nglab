"""
Tests for online normalization layer.
"""

import torch
from torch import nn

from python.src.pipeline.online_learning.normalization import OnlineNormalizer


class TestOnlineNormalizer:
    """Tests for the OnlineNormalizer class."""

    def test_init_default(self):
        normalizer = OnlineNormalizer(num_features=10)

        assert normalizer.num_features == 10
        assert normalizer.momentum is None
        assert normalizer.affine is True
        assert normalizer.eps == 1e-5

    def test_init_with_momentum(self):
        normalizer = OnlineNormalizer(num_features=5, momentum=0.1)

        assert normalizer.momentum == 0.1

    def test_init_without_affine(self):
        normalizer = OnlineNormalizer(num_features=5, affine=False)

        assert normalizer.weight is None
        assert normalizer.bias is None

    def test_init_buffers(self):
        normalizer = OnlineNormalizer(num_features=8)

        assert normalizer.running_mean.shape == (8,)
        assert normalizer.running_var.shape == (8,)
        assert torch.allclose(normalizer.running_mean, torch.zeros(8))
        assert torch.allclose(normalizer.running_var, torch.ones(8))
        assert normalizer.count.item() == 0.0

    def test_reset(self):
        normalizer = OnlineNormalizer(num_features=5)

        # Modify stats
        normalizer.running_mean.fill_(10.0)
        normalizer.running_var.fill_(2.0)
        normalizer.count.fill_(100.0)
        normalizer.weight.data.fill_(5.0)
        normalizer.bias.data.fill_(3.0)

        normalizer.reset()

        assert torch.allclose(normalizer.running_mean, torch.zeros(5))
        assert torch.allclose(normalizer.running_var, torch.ones(5))
        assert normalizer.count.item() == 0.0
        assert torch.allclose(normalizer.weight.data, torch.ones(5))
        assert torch.allclose(normalizer.bias.data, torch.zeros(5))

    def test_update_single_batch(self):
        normalizer = OnlineNormalizer(num_features=3)

        # Single batch
        x = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

        normalizer.update(x)

        # Mean should be [4, 5, 6]
        expected_mean = torch.tensor([4.0, 5.0, 6.0])
        assert torch.allclose(normalizer.running_mean, expected_mean)
        assert normalizer.count.item() == 3.0

    def test_update_multiple_batches_welford(self):
        normalizer = OnlineNormalizer(num_features=2, momentum=None)

        # First batch
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        normalizer.update(x1)

        # Second batch
        x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        normalizer.update(x2)

        # Combined mean should be [4, 5]
        expected_mean = torch.tensor([4.0, 5.0])
        assert torch.allclose(normalizer.running_mean, expected_mean)
        assert normalizer.count.item() == 4.0

    def test_update_with_momentum(self):
        normalizer = OnlineNormalizer(num_features=2, momentum=0.5)

        # First batch
        x1 = torch.tensor([[2.0, 4.0]])
        normalizer.update(x1)

        # With momentum 0.5:
        # new_mean = (1-0.5) * 0 + 0.5 * [2, 4] = [1, 2]
        expected_mean = torch.tensor([1.0, 2.0])
        assert torch.allclose(normalizer.running_mean, expected_mean)

        # Second batch
        x2 = torch.tensor([[6.0, 8.0]])
        normalizer.update(x2)

        # new_mean = (1-0.5) * [1, 2] + 0.5 * [6, 8] = [3.5, 5]
        expected_mean = torch.tensor([3.5, 5.0])
        assert torch.allclose(normalizer.running_mean, expected_mean)

    def test_update_1d_input(self):
        normalizer = OnlineNormalizer(num_features=3)

        # 1D input should be treated as batch size 1
        x = torch.tensor([10.0, 20.0, 30.0])
        normalizer.update(x)

        assert torch.allclose(normalizer.running_mean, x)
        assert normalizer.count.item() == 1.0

    def test_forward_normalizes_output(self):
        normalizer = OnlineNormalizer(num_features=2, affine=False)
        normalizer.eval()

        # Set known running stats
        normalizer.running_mean = torch.tensor([5.0, 10.0])
        normalizer.running_var = torch.tensor([4.0, 16.0])  # std = [2, 4]

        x = torch.tensor([[7.0, 18.0], [3.0, 2.0]])

        output = normalizer(x, update_stats=False)

        # Expected: (x - mean) / std
        # [7-5, 18-10] / [2, 4] = [1, 2]
        # [3-5, 2-10] / [2, 4] = [-1, -2]
        expected = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
        assert torch.allclose(output, expected, atol=1e-5)

    def test_forward_with_affine(self):
        normalizer = OnlineNormalizer(num_features=2, affine=True)
        normalizer.eval()

        # Set known stats
        normalizer.running_mean = torch.tensor([0.0, 0.0])
        normalizer.running_var = torch.tensor([1.0, 1.0])

        # Set affine parameters
        normalizer.weight.data = torch.tensor([2.0, 3.0])
        normalizer.bias.data = torch.tensor([1.0, -1.0])

        x = torch.tensor([[1.0, 2.0]])

        output = normalizer(x, update_stats=False)

        # norm_x = x (since mean=0, var=1)
        # output = norm_x * weight + bias = [1*2+1, 2*3-1] = [3, 5]
        expected = torch.tensor([[3.0, 5.0]])
        assert torch.allclose(output, expected, atol=1e-5)

    def test_forward_updates_stats_in_training(self):
        normalizer = OnlineNormalizer(num_features=2)
        normalizer.train()

        x = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

        _ = normalizer(x, update_stats=True)

        # Stats should be updated
        expected_mean = torch.tensor([20.0, 30.0])
        assert torch.allclose(normalizer.running_mean, expected_mean)
        assert normalizer.count.item() == 2.0

    def test_forward_no_update_in_eval(self):
        normalizer = OnlineNormalizer(num_features=2)
        normalizer.eval()

        initial_mean = normalizer.running_mean.clone()
        initial_count = normalizer.count.clone()

        x = torch.rand(10, 2)
        _ = normalizer(x, update_stats=True)  # update_stats=True but eval mode

        # In eval mode, update_stats is ignored (training=False)
        assert torch.allclose(normalizer.running_mean, initial_mean)
        assert normalizer.count == initial_count

    def test_forward_explicit_no_update(self):
        normalizer = OnlineNormalizer(num_features=2)
        normalizer.train()

        initial_mean = normalizer.running_mean.clone()

        x = torch.rand(10, 2)
        _ = normalizer(x, update_stats=False)

        # Even in training, explicit update_stats=False should not update
        assert torch.allclose(normalizer.running_mean, initial_mean)

    def test_gradient_flow(self):
        normalizer = OnlineNormalizer(num_features=3, affine=True)
        normalizer.eval()

        x = torch.randn(8, 3, requires_grad=True)

        output = normalizer(x, update_stats=False)
        loss = output.sum()
        loss.backward()

        # Gradients should flow through
        assert x.grad is not None
        assert normalizer.weight.grad is not None
        assert normalizer.bias.grad is not None

    def test_numerical_stability(self):
        normalizer = OnlineNormalizer(num_features=2, eps=1e-5)
        normalizer.eval()

        # Very small variance
        normalizer.running_mean = torch.tensor([0.0, 0.0])
        normalizer.running_var = torch.tensor([1e-10, 1e-10])

        x = torch.tensor([[1.0, 1.0]])

        output = normalizer(x, update_stats=False)

        # Should not produce inf/nan due to eps
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_integration_with_model(self):
        """Test OnlineNormalizer as part of a larger model."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.normalizer = OnlineNormalizer(5, affine=True)
                self.fc = nn.Linear(5, 2)

            def forward(self, x):
                x = self.normalizer(x)
                return self.fc(x)

        model = Model()
        model.train()

        x = torch.randn(16, 5)
        output = model(x)

        assert output.shape == (16, 2)

        # Optimizer step should work
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = output.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_large_batch_stability(self):
        normalizer = OnlineNormalizer(num_features=128)

        # Large batch
        x = torch.randn(1000, 128)
        normalizer.update(x)

        assert not torch.isnan(normalizer.running_mean).any()
        assert not torch.isnan(normalizer.running_var).any()
        assert (normalizer.running_var >= 0).all()
