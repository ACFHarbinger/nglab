import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from python.src.utils.logging.visualize_utils import (
    MyModelWrapper,
    get_batch,
    load_model_instance,
    log_weight_distributions,
    plot_logit_lens,
    plot_loss_landscape,
    plot_weight_trajectories,
    visualize_epoch,
)


@pytest.fixture
def dummy_model():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(12, 64)
            self.projection = nn.Linear(64, 2)
            # Mocking what plot_logit_lens expects: needs to handle (x, x_mark)
            # return shape [1, 10, 64] to match projection input
            self.enc_embedding = MagicMock(
                side_effect=lambda x, m: torch.randn(x.size(0), x.size(1), 64)
            )

        def forward(self, x):
            return self.projection(torch.relu(self.encoder(x)))

    return SimpleModel()


def test_get_batch():
    batch = get_batch(torch.device("cpu"), seq_len=10, batch_size=8, feature_dim=5)
    assert batch.shape == (8, 10, 5)


def test_my_model_wrapper(dummy_model):
    wrapper = MyModelWrapper(dummy_model)
    x = torch.randn(2, 12)  # Direct input to linear
    out = wrapper(x)
    assert out.shape == (2, 2)


@patch("python.src.utils.logging.visualize_utils.load_model")
def test_load_model_instance(mock_load, dummy_model):
    mock_load.return_value = (dummy_model, {"opt": 1})
    model, opts = load_model_instance("fake_path", torch.device("cpu"))
    assert model == dummy_model
    assert opts["opt"] == 1
    assert not model.training


def test_plot_weight_trajectories(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    output_file = tmp_path / "trajectory.png"

    # Create dummy checkpoints
    for i in range(3):
        cp_path = checkpoint_dir / f"model-epoch{i}.pt"
        torch.save({"model": {"w": torch.randn(10)}}, cp_path)

    plot_weight_trajectories(str(checkpoint_dir), str(output_file))
    assert output_file.exists()


def test_log_weight_distributions(dummy_model, tmp_path):
    log_dir = tmp_path / "logs"
    log_weight_distributions(dummy_model, epoch=1, log_dir=str(log_dir))
    # Check if events file exists
    assert os.path.isdir(log_dir)
    assert any(f.startswith("events.out.tfevents") for f in os.listdir(log_dir))


def test_plot_logit_lens(dummy_model, tmp_path):
    output_file = tmp_path / "logit_lens.png"
    x_batch = torch.randn(1, 10, 12)

    # Needs some more mocking in dummy_model to satisfy the manual traversal
    # Layer Index (0=Embed, 1..N=Encoder)
    class MockLayer(nn.Module):
        def forward(self, x):
            return x, None

    # The code expects encoder.attn_layers
    dummy_model.encoder.attn_layers = nn.ModuleList([MockLayer()])

    plot_logit_lens(dummy_model, x_batch, str(output_file))
    assert output_file.exists()


def test_plot_loss_landscape(dummy_model, tmp_path):
    output_dir = tmp_path / "landscapes"
    opts = {"device": "cpu"}

    # loss-landscapes might be tricky to mock, let's try running it with very low resolution
    plot_loss_landscape(dummy_model, opts, str(output_dir), resolution=2, span=0.1)

    # Check if file exists (it appends epoch)
    files = os.listdir(output_dir)
    assert any("landscape_2d_epoch_0.png" in f for f in files)


def test_visualize_epoch(dummy_model, tmp_path):
    opts = {
        "viz_modes": ["distributions", "trajectory"],
        "log_dir": str(tmp_path / "logs"),
        "save_dir": str(tmp_path / "checkpoints"),
        "run_name": "test_run",
    }
    # Create checkpoints dir for trajectory
    os.makedirs(opts["save_dir"], exist_ok=True)
    cp_path = os.path.join(opts["save_dir"], "model-epoch0.pt")
    torch.save({"model": {"w": torch.randn(10)}}, cp_path)
    # Another one
    cp_path2 = os.path.join(opts["save_dir"], "model-epoch1.pt")
    torch.save({"model": {"w": torch.randn(10)}}, cp_path2)

    visualize_epoch(dummy_model, opts, epoch=1)

    viz_dir = os.path.join(opts["log_dir"], "visualizations")
    assert os.path.exists(os.path.join(viz_dir, "trajectory.png"))
