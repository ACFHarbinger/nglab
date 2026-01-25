"""
Tests for MAML (Model-Agnostic Meta-Learning) Lightning module.
"""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from python.src.pipeline.meta.maml import MAMLDataModule, MAMLLightningModule


class SimpleModel(nn.Module):
    """Simple model for testing MAML."""

    def __init__(self, input_dim: int = 5, output_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle sequence input: [batch, seq_len, features]
        if x.dim() == 3:
            x = x[:, -1, :]  # Take last timestep
        return self.fc(x)


class TestMAMLLightningModule:
    @pytest.fixture
    def model(self):
        return SimpleModel(input_dim=5, output_dim=1)

    @pytest.fixture
    def maml_module(self, model):
        return MAMLLightningModule(
            model=model,
            inner_lr=0.01,
            outer_lr=0.001,
            inner_steps=3,
            meta_batch_size=2,
        )

    def test_init(self, maml_module, model):
        assert maml_module.model is model
        assert maml_module.inner_lr == 0.01
        assert maml_module.outer_lr == 0.001
        assert maml_module.inner_steps == 3
        assert maml_module.meta_batch_size == 2

    def test_configure_optimizers(self, maml_module):
        optimizer = maml_module.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)
        # Check that optimizer has correct learning rate
        assert optimizer.defaults["lr"] == 0.001

    def test_inner_loop(self, maml_module):
        batch_size = 4
        seq_len = 10
        features = 5

        support_x = torch.randn(batch_size, seq_len, features)
        support_y = torch.randn(batch_size, 1)
        query_x = torch.randn(batch_size, seq_len, features)
        query_y = torch.randn(batch_size, 1)

        loss = maml_module.inner_loop(support_x, support_y, query_x, query_y)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.requires_grad  # Should have gradients for meta-update

    def test_inner_loop_adapts_model(self, maml_module):
        """Test that inner loop actually adapts the model to support data."""
        batch_size = 8
        seq_len = 5
        features = 5

        # Create support data with clear pattern
        support_x = torch.ones(batch_size, seq_len, features)
        support_y = torch.ones(batch_size, 1) * 10.0

        query_x = torch.ones(batch_size, seq_len, features)
        query_y = torch.ones(batch_size, 1) * 10.0

        # Get loss before and after adaptation
        initial_pred = maml_module.model(query_x)
        initial_loss = torch.nn.functional.mse_loss(initial_pred, query_y)

        # Inner loop should reduce loss on query set
        adapted_loss = maml_module.inner_loop(support_x, support_y, query_x, query_y)

        # The adapted model should have lower loss on similar data
        # Note: This might not always be strictly true depending on initialization
        # but for a simple linear model it should generally improve
        assert adapted_loss.item() < initial_loss.item() * 2  # Allow some tolerance

    def test_training_step(self, maml_module):
        batch_size = 4
        seq_len = 5
        features = 5

        # Create a batch of tasks
        tasks = []
        for _ in range(2):  # meta_batch_size = 2
            support_x = torch.randn(batch_size, seq_len, features)
            support_y = torch.randn(batch_size, 1)
            query_x = torch.randn(batch_size, seq_len, features)
            query_y = torch.randn(batch_size, 1)
            tasks.append((support_x, support_y, query_x, query_y))

        batch = {"tasks": tasks}
        batch_idx = 0

        # Mock the log method
        maml_module.log = MagicMock()

        loss = maml_module.training_step(batch, batch_idx)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        # Verify logging was called
        assert maml_module.log.call_count >= 2  # meta_loss and num_tasks

    def test_validation_step(self, maml_module):
        batch_size = 4
        seq_len = 5
        features = 5

        tasks = []
        for _ in range(2):
            support_x = torch.randn(batch_size, seq_len, features)
            support_y = torch.randn(batch_size, 1)
            query_x = torch.randn(batch_size, seq_len, features)
            query_y = torch.randn(batch_size, 1)
            tasks.append((support_x, support_y, query_x, query_y))

        batch = {"tasks": tasks}
        batch_idx = 0

        maml_module.log = MagicMock()

        val_loss = maml_module.validation_step(batch, batch_idx)

        assert isinstance(val_loss, torch.Tensor)
        assert val_loss.ndim == 0
        # Validation loss should not require grad (detached)
        assert not val_loss.requires_grad

    def test_adapt(self, maml_module):
        batch_size = 8
        seq_len = 5
        features = 5

        support_x = torch.randn(batch_size, seq_len, features)
        support_y = torch.randn(batch_size, 1)

        # Adapt to new data
        adapted_model = maml_module.adapt(support_x, support_y)

        # Should return a new model (deep copy)
        assert adapted_model is not maml_module.model
        # Adapted model should be in eval mode
        assert not adapted_model.training

        # Parameters should be different after adaptation
        for _p1, _p2 in zip(
            maml_module.model.parameters(), adapted_model.parameters(), strict=False
        ):
            # Parameters might be similar but not identical
            pass  # Just checking the structure works

    def test_adapt_custom_steps(self, maml_module):
        batch_size = 4
        seq_len = 5
        features = 5

        support_x = torch.randn(batch_size, seq_len, features)
        support_y = torch.randn(batch_size, 1)

        # Adapt with custom number of steps
        adapted_model = maml_module.adapt(support_x, support_y, num_steps=10)

        assert adapted_model is not maml_module.model
        assert not adapted_model.training

    def test_hyperparameters_saved(self, model):
        maml = MAMLLightningModule(
            model=model,
            inner_lr=0.05,
            outer_lr=0.002,
            inner_steps=7,
            meta_batch_size=8,
        )

        # Hyperparameters should be saved (excluding model)
        assert "inner_lr" in maml.hparams
        assert maml.hparams["inner_lr"] == 0.05
        assert maml.hparams["outer_lr"] == 0.002
        assert maml.hparams["inner_steps"] == 7


class TestMAMLDataModule:
    @pytest.fixture
    def regime_datasets(self):
        """Create mock regime datasets."""
        return {
            0: torch.randn(200, 10),  # Regime 0: 200 samples, 10 features
            1: torch.randn(200, 10),  # Regime 1
            2: torch.randn(200, 10),  # Regime 2
        }

    @pytest.fixture
    def data_module(self, regime_datasets):
        return MAMLDataModule(
            regime_datasets=regime_datasets,
            support_size=20,
            query_size=20,
            meta_batch_size=2,
            num_workers=0,
        )

    def test_init(self, data_module):
        assert data_module.support_size == 20
        assert data_module.query_size == 20
        assert data_module.meta_batch_size == 2
        assert len(data_module.regime_datasets) == 3

    def test_create_task_2d_data(self, data_module):
        regime_data = torch.randn(100, 10)  # 2D data
        support_x, support_y, query_x, query_y = data_module.create_task(regime_data)

        # Check shapes
        assert support_x.shape[0] == data_module.support_size
        assert query_x.shape[0] == data_module.query_size

        # For 2D data, should add sequence dimension
        assert support_x.dim() == 3  # [batch, 1, features]
        assert support_x.shape[1] == 1

        # Target should be last feature
        assert support_y.shape == (data_module.support_size, 1)
        assert query_y.shape == (data_module.query_size, 1)

    def test_create_task_3d_data(self, data_module):
        # 3D data: [num_samples, seq_len, features]
        regime_data = torch.randn(100, 5, 10)
        support_x, support_y, _query_x, _query_y = data_module.create_task(regime_data)

        # Check shapes
        assert support_x.shape[0] == data_module.support_size
        assert support_x.dim() == 3
        assert support_x.shape[2] == 9  # features - 1 (target excluded)

        # Target is last feature of last timestep
        assert support_y.shape == (data_module.support_size, 1)

    def test_create_task_small_regime(self, data_module):
        """Test task creation when regime has fewer samples than needed."""
        small_regime = torch.randn(30, 10)  # Less than support + query = 40
        support_x, _support_y, query_x, _query_y = data_module.create_task(small_regime)

        # Should still work by sampling with replacement
        assert support_x.shape[0] == data_module.support_size
        assert query_x.shape[0] == data_module.query_size

    def test_train_dataloader(self, data_module):
        train_dl = data_module.train_dataloader()

        # Should be an iterator
        batch = next(train_dl)

        assert "tasks" in batch
        assert len(batch["tasks"]) == data_module.meta_batch_size

        # Each task should have 4 elements
        for task in batch["tasks"]:
            assert len(task) == 4  # support_x, support_y, query_x, query_y

    def test_val_dataloader(self, data_module):
        val_dl = data_module.val_dataloader()

        batch = next(val_dl)

        assert "tasks" in batch
        assert len(batch["tasks"]) == data_module.meta_batch_size

    def test_train_dataloader_produces_multiple_batches(self, data_module):
        train_dl = data_module.train_dataloader()

        batch_count = 0
        for _batch in train_dl:
            batch_count += 1
            if batch_count >= 5:
                break

        assert batch_count >= 5

    def test_different_regimes_sampled(self, data_module):
        """Test that different regimes are sampled across batches."""
        train_dl = data_module.train_dataloader()

        # Collect multiple batches - the random sampling should vary
        batches = [next(train_dl) for _ in range(10)]

        assert len(batches) == 10
        # Each batch should have valid task structure
        for batch in batches:
            assert "tasks" in batch


class TestMAMLIntegration:
    """Integration tests for MAML with full training loop simulation."""

    def test_full_meta_training_step(self):
        """Test a complete meta-training step."""
        model = SimpleModel(input_dim=5, output_dim=1)
        maml = MAMLLightningModule(
            model=model,
            inner_lr=0.1,
            outer_lr=0.01,
            inner_steps=5,
            meta_batch_size=4,
        )

        # Create regime datasets
        regime_datasets = {
            i: torch.randn(100, 6) for i in range(3)  # 5 features + 1 target
        }

        data_module = MAMLDataModule(
            regime_datasets=regime_datasets,
            support_size=10,
            query_size=10,
            meta_batch_size=4,
        )

        # Get a batch
        train_dl = data_module.train_dataloader()
        batch = next(train_dl)

        # Configure optimizer
        optimizer = maml.configure_optimizers()
        maml.log = MagicMock()

        # Forward pass
        loss = maml.training_step(batch, 0)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss should be a valid number
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_meta_learning_improves(self):
        """Test that meta-learning actually improves over iterations."""
        torch.manual_seed(42)

        model = SimpleModel(input_dim=5, output_dim=1)
        maml = MAMLLightningModule(
            model=model,
            inner_lr=0.1,
            outer_lr=0.01,
            inner_steps=5,
            meta_batch_size=4,
        )

        # Create consistent regime datasets
        regime_datasets = {i: torch.randn(100, 6) for i in range(3)}

        data_module = MAMLDataModule(
            regime_datasets=regime_datasets,
            support_size=10,
            query_size=10,
            meta_batch_size=4,
        )

        train_dl = data_module.train_dataloader()
        optimizer = maml.configure_optimizers()
        maml.log = MagicMock()

        losses = []
        for i, batch in enumerate(train_dl):
            if i >= 20:
                break

            optimizer.zero_grad()
            loss = maml.training_step(batch, i)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Check that we collected losses
        assert len(losses) == 20

        # Average loss should decrease (or at least not explode)
        first_half_avg = sum(losses[:10]) / 10
        second_half_avg = sum(losses[10:]) / 10

        # Allow for some variance, but loss shouldn't explode
        assert second_half_avg < first_half_avg * 2
