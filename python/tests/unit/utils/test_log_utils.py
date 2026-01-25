import json
import os
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from python.src.utils.logging.log_utils import (
    _convert_numpy,
    log_epoch,
    log_timeseries_values,
    log_to_json_resilient,
    plot_training_results,
)


@pytest.fixture
def mock_wandb():
    with patch("python.src.utils.logging.log_utils.wandb") as mock:
        yield mock


@pytest.fixture
def mock_plt():
    with patch("python.src.utils.logging.log_utils.plt") as mock:
        yield mock


def test_log_timeseries_values(mock_wandb):
    tb_logger = MagicMock()
    opts = {"log_step": 1, "wandb_mode": "online"}
    output = torch.randn(2, 5)
    grad_norms = ([1.0], [0.5])

    log_timeseries_values(
        loss=0.1,
        grad_norms=grad_norms,
        epoch=1,
        batch_id=1,
        step=10,
        output=output,
        tb_logger=tb_logger,
        opts=opts,
    )

    tb_logger.log_value.assert_any_call("loss", 0.1, 10)
    mock_wandb.log.assert_called_once()
    assert mock_wandb.log.call_args[0][0]["train/loss"] == 0.1


def test_log_epoch(mock_wandb):
    optimizer = MagicMock()
    optimizer.param_groups = [{"lr": 0.001}]
    opts = {"wandb_mode": "online"}

    log_epoch(
        epoch=1, epoch_duration=120.0, optimizer=optimizer, opts=opts, avg_loss=0.05
    )

    mock_wandb.log.assert_called_once()
    args = mock_wandb.log.call_args[0][0]
    assert args["epoch/avg_loss"] == 0.05
    assert args["epoch/lr"] == 0.001


def test_plot_training_results(mock_plt, tmp_path):
    df = pd.DataFrame({"loss": [0.5, 0.4, 0.3]})
    save_path = str(tmp_path / "plot.png")

    plot_training_results(df, save_path)

    mock_plt.savefig.assert_called_with(save_path)
    mock_plt.close.assert_called_once()


def test_convert_numpy():
    data = {
        "arr": np.array([1, 2]),
        "int": np.int64(10),
        "float": np.float32(0.5),
        "nested": [{"a": np.array([3])}],
    }
    converted = _convert_numpy(data)

    assert isinstance(converted["arr"], list)
    assert isinstance(converted["int"], int)
    assert isinstance(converted["float"], float)
    assert converted["nested"][0]["a"] == [3]


def test_log_to_json_resilient(tmp_path):
    json_path = str(tmp_path / "log.json")
    data1 = {"step": 1, "val": 0.5}

    # First write
    log_to_json_resilient(json_path, data1)

    with open(json_path) as f:
        loaded = json.load(f)
    assert loaded == data1

    # Second write (merge)
    data2 = {"step": 2, "val": 0.6}
    log_to_json_resilient(json_path, data2)

    with open(json_path) as f:
        loaded = json.load(f)
    assert loaded["step"] == 2  # Merged/Overwritten
    assert loaded["val"] == 0.6


def test_log_to_json_resilient_error_fallback(tmp_path):
    json_path = str(tmp_path / "protected.json")
    # Make it a directory to cause write error
    os.makedirs(json_path)

    data = {"test": "data"}
    # This should trigger the fallback logic
    with patch("builtins.print") as mock_print:
        log_to_json_resilient(json_path, data)
        mock_print.assert_any_call(ANY)  # Should print error msg

    # Check if a fallback file was created in the same dir?
    # Actually json_path is a dir, so fallback_path = f"{json_path}.{timestamp}.bak" might fail too if timestamp contains / or if it tries to write to the dir.
    # But here it's f"protected.json.timestamp.bak"
    files = os.listdir(tmp_path)
    assert any(".bak" in f for f in files)
