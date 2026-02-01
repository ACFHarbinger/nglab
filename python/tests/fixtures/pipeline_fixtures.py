from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_trainer() -> MagicMock:
    """Mock PyTorch Lightning Trainer."""
    trainer = MagicMock()
    trainer.fit = MagicMock()
    trainer.test = MagicMock()
    return trainer

@pytest.fixture
def mock_lightning_module() -> MagicMock:
    """Mock PyTorch Lightning Module."""
    module = MagicMock()
    module.training_step = MagicMock(return_value={"loss": 1.0})
    module.validation_step = MagicMock()
    module.configure_optimizers = MagicMock()
    return module

@pytest.fixture
def mock_callback() -> MagicMock:
    """Mock Training Callback."""
    callback = MagicMock()
    callback.on_train_batch_end = MagicMock()
    callback.on_validation_epoch_end = MagicMock()
    return callback

__all__ = ["mock_callback", "mock_lightning_module", "mock_trainer"]
