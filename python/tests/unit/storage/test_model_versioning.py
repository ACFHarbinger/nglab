"""
Unit tests for model versioning module.

Tests ModelMetadata serialization, model save/load operations,
ModelRegistry functionality, and version compatibility checking.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from python.src.utils.io.model_versioning import (
    ModelMetadata,
    ModelRegistry,
    check_version_compatibility,
    compute_dataset_hash,
    get_git_commit,
    load_model_with_metadata,
    save_model_with_metadata,
)

# Test fixtures


@pytest.fixture
def simple_model():
    """Create a simple test model."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 1)

        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x)))

    return SimpleModel()


@pytest.fixture
def sample_metadata():
    """Create sample model metadata."""
    return ModelMetadata(
        version="1.0.0",
        model_type="test_model",
        framework_version="2.0.0",
        description="Test model for unit testing",
        dataset_hash="abc123",
        training_config={"lr": 0.001, "batch_size": 32},
        metrics={"val_loss": 0.123, "val_acc": 0.95},
        git_commit="a1b2c3d4",
        dependencies={"torch": "2.0.0", "numpy": "1.24.0"},
    )


# ============================================================
# ModelMetadata Tests
# ============================================================


def test_metadata_to_dict(sample_metadata):
    """Test converting ModelMetadata to dictionary."""
    metadata_dict = sample_metadata.to_dict()

    assert isinstance(metadata_dict, dict)
    assert metadata_dict["version"] == "1.0.0"
    assert metadata_dict["model_type"] == "test_model"
    assert metadata_dict["training_config"]["lr"] == 0.001
    assert metadata_dict["metrics"]["val_loss"] == 0.123


def test_metadata_from_dict(sample_metadata):
    """Test creating ModelMetadata from dictionary."""
    metadata_dict = sample_metadata.to_dict()
    reconstructed = ModelMetadata.from_dict(metadata_dict)

    assert reconstructed.version == sample_metadata.version
    assert reconstructed.model_type == sample_metadata.model_type
    assert reconstructed.description == sample_metadata.description
    assert reconstructed.dataset_hash == sample_metadata.dataset_hash


def test_metadata_to_json(sample_metadata):
    """Test converting ModelMetadata to JSON string."""
    json_str = sample_metadata.to_json()

    assert isinstance(json_str, str)
    # Verify it's valid JSON
    parsed = json.loads(json_str)
    assert parsed["version"] == "1.0.0"
    assert parsed["model_type"] == "test_model"


def test_metadata_from_json(sample_metadata):
    """Test creating ModelMetadata from JSON string."""
    json_str = sample_metadata.to_json()
    reconstructed = ModelMetadata.from_json(json_str)

    assert reconstructed.version == sample_metadata.version
    assert reconstructed.model_type == sample_metadata.model_type
    assert reconstructed.training_config == sample_metadata.training_config


def test_metadata_roundtrip(sample_metadata):
    """Test full serialization roundtrip (dict and JSON)."""
    # Dict roundtrip
    dict_roundtrip = ModelMetadata.from_dict(sample_metadata.to_dict())
    assert dict_roundtrip.version == sample_metadata.version
    assert dict_roundtrip.metrics == sample_metadata.metrics

    # JSON roundtrip
    json_roundtrip = ModelMetadata.from_json(sample_metadata.to_json())
    assert json_roundtrip.version == sample_metadata.version
    assert json_roundtrip.dependencies == sample_metadata.dependencies


# ============================================================
# Save/Load Model Tests
# ============================================================


def test_save_model_with_metadata(simple_model, sample_metadata, tmp_path):
    """Test saving model checkpoint with metadata."""
    save_path = tmp_path / "test_checkpoint.pt"

    save_model_with_metadata(simple_model, save_path, sample_metadata)

    # Verify file was created
    assert save_path.exists()

    # Load and verify contents
    checkpoint = torch.load(save_path, weights_only=False, map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert "metadata" in checkpoint
    assert checkpoint["metadata"]["version"] == "1.0.0"


def test_save_model_with_optimizer(simple_model, sample_metadata, tmp_path):
    """Test saving model with optimizer state."""
    save_path = tmp_path / "test_checkpoint.pt"
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)

    # Take a step to have non-empty optimizer state
    loss = simple_model(torch.randn(2, 10)).sum()
    loss.backward()
    optimizer.step()

    save_model_with_metadata(
        simple_model, save_path, sample_metadata, optimizer=optimizer
    )

    # Load and verify
    checkpoint = torch.load(save_path, weights_only=False, map_location="cpu")
    assert "optimizer_state_dict" in checkpoint
    assert len(checkpoint["optimizer_state_dict"]) > 0


def test_load_model_with_metadata(simple_model, sample_metadata, tmp_path):
    """Test loading model checkpoint with metadata."""
    save_path = tmp_path / "test_checkpoint.pt"

    # Save first
    save_model_with_metadata(simple_model, save_path, sample_metadata)

    # Create a new model instance
    new_model = type(simple_model)()

    # Load
    loaded_model, loaded_metadata = load_model_with_metadata(
        new_model, save_path, map_location="cpu"
    )

    # Verify metadata
    assert loaded_metadata.version == "1.0.0"
    assert loaded_metadata.model_type == "test_model"

    # Verify model weights match
    for param1, param2 in zip(
        simple_model.parameters(), loaded_model.parameters(), strict=False
    ):
        assert torch.allclose(param1, param2)


def test_load_model_strict_mode(simple_model, sample_metadata, tmp_path):
    """Test loading model with strict mode enabled."""
    save_path = tmp_path / "test_checkpoint.pt"

    # Save model
    save_model_with_metadata(simple_model, save_path, sample_metadata)

    # Try loading into incompatible model (should fail in strict mode)
    class IncompatibleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            # Missing linear2

    incompatible_model = IncompatibleModel()

    # This should raise an error with strict=True
    with pytest.raises(RuntimeError):
        load_model_with_metadata(incompatible_model, save_path, strict=True)


# ============================================================
# Version Compatibility Tests
# ============================================================


def test_version_compatibility_same():
    """Test version compatibility with same version."""
    assert check_version_compatibility("1.0.0", "1.0.0") is True


def test_version_compatibility_minor():
    """Test version compatibility with different minor version."""
    # Minor version mismatch should still be compatible with warning
    assert check_version_compatibility("1.1.0", "1.0.0") is True
    assert check_version_compatibility("1.0.0", "1.1.0") is True


def test_version_compatibility_patch():
    """Test version compatibility with different patch version."""
    # Patch version mismatch should be compatible
    assert check_version_compatibility("1.0.1", "1.0.0") is True
    assert check_version_compatibility("1.0.0", "1.0.1") is True


def test_version_compatibility_major():
    """Test version compatibility with different major version."""
    # Major version mismatch should be incompatible
    assert check_version_compatibility("2.0.0", "1.0.0") is False
    assert check_version_compatibility("1.0.0", "2.0.0") is False


# ============================================================
# ModelRegistry Tests
# ============================================================


def test_model_registry_initialization(tmp_path):
    """Test initializing ModelRegistry."""
    registry = ModelRegistry(tmp_path)

    assert registry.base_path == Path(tmp_path)
    assert registry.base_path.exists()


def test_model_registry_save(simple_model, sample_metadata, tmp_path):
    """Test saving model to registry."""
    registry = ModelRegistry(tmp_path)

    save_path = registry.save(
        simple_model, model_type="test_model", version="1.0.0", metadata=sample_metadata
    )

    assert save_path.exists()
    assert "test_model" in str(save_path)
    assert "1.0.0" in str(save_path)


def test_model_registry_load(simple_model, sample_metadata, tmp_path):
    """Test loading model from registry."""
    registry = ModelRegistry(tmp_path)

    # Save first
    registry.save(
        simple_model, model_type="test_model", version="1.0.0", metadata=sample_metadata
    )

    # Load
    new_model = type(simple_model)()
    loaded_model, loaded_metadata = registry.load(
        new_model, model_type="test_model", version="1.0.0"
    )

    # Verify
    assert loaded_metadata.version == "1.0.0"
    for param1, param2 in zip(
        simple_model.parameters(), loaded_model.parameters(), strict=False
    ):
        assert torch.allclose(param1, param2)


def test_model_registry_list_versions(simple_model, sample_metadata, tmp_path):
    """Test listing available versions."""
    registry = ModelRegistry(tmp_path)

    # Save multiple versions
    for i in range(3):
        meta = ModelMetadata(
            version=f"1.0.{i}",
            model_type="test_model",
            training_config={},
            metrics={},
            description=f"Version 1.0.{i}",
        )
        registry.save(
            simple_model, model_type="test_model", version=f"1.0.{i}", metadata=meta
        )

    # List versions
    versions = registry.list_versions("test_model")

    assert len(versions) == 3
    assert "1.0.0" in versions
    assert "1.0.1" in versions
    assert "1.0.2" in versions


def test_model_registry_get_latest(simple_model, sample_metadata, tmp_path):
    """Test getting latest version."""
    registry = ModelRegistry(tmp_path)

    # Save multiple versions
    versions = ["1.0.0", "1.1.0", "1.2.0"]
    for version in versions:
        meta = ModelMetadata(
            version=version,
            model_type="test_model",
            training_config={},
            metrics={},
            description=f"Version {version}",
        )
        registry.save(
            simple_model, model_type="test_model", version=version, metadata=meta
        )

    # Get latest
    latest = registry.get_latest("test_model")

    # Should be "1.2.0" (highest version)
    assert latest == "1.2.0"


def test_model_registry_delete(simple_model, sample_metadata, tmp_path):
    """Test deleting a model version."""
    registry = ModelRegistry(tmp_path)

    # Save
    registry.save(
        simple_model, model_type="test_model", version="1.0.0", metadata=sample_metadata
    )

    # Verify exists
    assert len(registry.list_versions("test_model")) == 1

    # Delete
    result = registry.delete("test_model", "1.0.0")
    assert result is True

    # Verify deleted
    assert len(registry.list_versions("test_model")) == 0


def test_model_registry_get_metadata(simple_model, sample_metadata, tmp_path):
    """Test getting metadata without loading model."""
    registry = ModelRegistry(tmp_path)

    # Save
    registry.save(
        simple_model, model_type="test_model", version="1.0.0", metadata=sample_metadata
    )

    # Get metadata
    metadata = registry.get_metadata("test_model", "1.0.0")

    assert metadata.version == "1.0.0"
    assert metadata.model_type == "test_model"
    assert metadata.metrics == sample_metadata.metrics


# ============================================================
# Utility Function Tests
# ============================================================


def test_compute_dataset_hash(tmp_path):
    """Test computing dataset hash."""
    # Create a test file
    test_file = tmp_path / "test_data.txt"
    test_file.write_text("test data content")

    hash_value = compute_dataset_hash(test_file)

    assert isinstance(hash_value, str)
    assert len(hash_value) == 64  # SHA256 produces 64 hex characters

    # Hash should be deterministic
    hash_value2 = compute_dataset_hash(test_file)
    assert hash_value == hash_value2


@patch("subprocess.run")
def test_get_git_commit(mock_run):
    """Test getting git commit hash."""
    # Mock successful git command
    mock_run.return_value = MagicMock(stdout="a1b2c3d4e5f6\n", stderr="", returncode=0)

    commit = get_git_commit()

    assert commit == "a1b2c3d4e5f6"
    mock_run.assert_called_once()


@patch("subprocess.run")
def test_get_git_commit_not_repo(mock_run):
    """Test getting git commit when not in a repo."""
    # Mock failed git command
    from subprocess import CalledProcessError

    mock_run.side_effect = CalledProcessError(1, ["git", "rev-parse", "HEAD"])

    commit = get_git_commit()

    assert commit is None
