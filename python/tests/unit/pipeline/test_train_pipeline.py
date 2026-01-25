from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from python.src.pipeline.train import clip_grad_norms, rollout, train_batch, train_epoch


@pytest.fixture
def dummy_model():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            # x is a dict in rollout
            if isinstance(x, dict):
                batch_size = x["Price"].size(0)
            else:
                batch_size = x.size(0)

            if self.training:
                return self.fc(x if not isinstance(x, dict) else x["Price"])
            else:
                return torch.zeros(batch_size), None

    return SimpleModel()


@pytest.fixture
def dummy_dataset():
    class IterDataset(torch.utils.data.Dataset):
        def __init__(self, size=10):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, i):
            return {"Price": torch.randn(10), "Labels": torch.randn(2)}

    return IterDataset()


def test_rollout(dummy_model, dummy_dataset):
    opts = {"device": "cpu", "eval_batch_size": 2, "no_progress_bar": True}
    results = rollout(dummy_model, dummy_dataset, opts)

    assert results.shape == (10,)
    assert not dummy_model.training


def test_clip_grad_norms(dummy_model):
    optimizer = torch.optim.Adam(dummy_model.parameters())
    # Add dummy gradients
    for p in dummy_model.parameters():
        p.grad = torch.randn_like(p)

    norms, clipped = clip_grad_norms(optimizer.param_groups, max_norm=0.1)

    assert len(norms) == 1
    assert clipped[0] <= 0.1


@patch("python.src.pipeline.train.train_batch")
@patch("python.src.pipeline.train.log_epoch")
@patch("python.src.pipeline.train.get_inner_model")
@patch("torch.save")
def test_train_epoch(  # noqa: PLR0913
    mock_torch_save,
    mock_get_inner,
    mock_log_epoch,
    mock_train_batch,
    dummy_model,
    dummy_dataset,
    tmp_path,
):
    """Test train_epoch function."""
    optimizer = torch.optim.Adam(dummy_model.parameters())
    baseline = MagicMock()
    scheduler = MagicMock()
    tb_logger = MagicMock()
    opts = {
        "run_name": "test",
        "no_tensorboard": False,
        "batch_size": 2,
        "no_progress_bar": True,
        "checkpoint_epochs": 1,
        "save_dir": str(tmp_path),
        "n_epochs": 1,
    }

    mock_get_inner.return_value = dummy_model

    train_epoch(
        dummy_model, optimizer, baseline, scheduler, 0, dummy_dataset, tb_logger, opts
    )

    # Dataset size 10, batch size 2 -> 5 batches
    assert mock_train_batch.call_count == 5
    mock_log_epoch.assert_called_once()
    scheduler.step.assert_called_once()

    # Check if checkpoint saved
    mock_torch_save.assert_called_once()
    args, _kwargs = mock_torch_save.call_args
    assert "model" in args[0]
    assert "optimizer" in args[0]


@patch("python.src.pipeline.train.log_timeseries_values")
def test_train_batch(mock_log, dummy_model):
    optimizer = torch.optim.Adam(dummy_model.parameters())
    baseline = MagicMock()
    tb_logger = MagicMock()
    batch = {"Price": torch.randn(2, 10), "Labels": torch.randn(2, 2)}
    opts = {"device": "cpu", "max_grad_norm": 1.0, "log_step": 1}

    train_batch(dummy_model, optimizer, baseline, 0, 0, batch, 0, tb_logger, opts)

    assert mock_log.call_count == 1
    mock_log.assert_called_once()
