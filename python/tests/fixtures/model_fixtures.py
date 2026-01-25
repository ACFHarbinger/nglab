import pytest
import torch

from python.src.models.time_series import TimeSeriesBackbone
from python.src.utils.io.model_versioning import ModelMetadata


@pytest.fixture
def mock_model_config():
    return {
        "name": "LSTM",
        "feature_dim": 1,
        "output_dim": 1,
        "hidden_dim": 16,
        "num_layers": 1,
        "dropout": 0.0,
        "normalization": {"method": "minmax", "min": 0.0, "max": 100.0},
    }


@pytest.fixture
def mock_model_artifact(tmp_path, mock_model_config):
    """Creates a dummy model checkpoint and metadata file in a temp directory."""
    # Create Metadata
    metadata = ModelMetadata(
        version="0.1.0",
        model_type="lstm_test",
        framework_version="pytorch-2.0",
        training_config={
            "model": mock_model_config,
            "normalization": mock_model_config["normalization"],
        },
        metrics={"val_loss": 0.05},
        training_date="2024-01-01T00:00:00",
        dataset_hash="unknown_hash",
        git_commit="test_hash",
    )

    # Initialize Model
    # TimeSeriesBackbone expects a config dict with specific keys.
    # We should ensure the config matches what the class expects.
    # Assuming TimeSeriesBackbone handles generic args or we pass minimal required.
    model = TimeSeriesBackbone(mock_model_config)

    # Save .pt (checkpoint)
    model_path = tmp_path / "test_model_checkpoint.pt"
    # Saving both state_dict and metadata inside checkpoint as fallback
    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": metadata.to_dict()},
        model_path,
    )

    # Save .json sidecar
    meta_path = model_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        f.write(metadata.to_json())

    return model_path
