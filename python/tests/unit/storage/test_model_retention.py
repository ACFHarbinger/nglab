from unittest.mock import MagicMock

import pytest

from python.src.utils.io.model_retention import ModelRetentionPolicy, RetentionConfig


@pytest.fixture
def mock_manager():
    manager = MagicMock()
    manager.list_versions.return_value = [
        "1.0.0",
        "1.1.0",
        "1.2.0",
        "1.3.0",
        "1.4.0",
        "1.5.0",
    ]
    return manager


def test_retention_config():
    config = RetentionConfig(keep_latest_n=3, dry_run=True)
    assert config.keep_latest_n == 3
    assert config.dry_run is True


def test_model_retention_cleanup(mock_manager):
    config = RetentionConfig(keep_latest_n=3)
    policy = ModelRetentionPolicy(config, mock_manager)

    # We have 6 versions, keep 3 -> delete 3
    deleted_count = policy.cleanup_versions("ppo")

    assert deleted_count == 3
    # Check if delete was called for the oldest versions
    # Sorted versions: 1.0.0, 1.1.0, 1.2.0, 1.3.0, 1.4.0, 1.5.0
    # Keep latest 3: 1.3.0, 1.4.0, 1.5.0
    # Delete: 1.0.0, 1.1.0, 1.2.0

    # Depending on which delete method the mock has
    # mock_manager is generic, let's give it delete_checkpoint (like CloudManager)
    mock_manager.delete_checkpoint.assert_any_call("ppo", "1.0.0")
    mock_manager.delete_checkpoint.assert_any_call("ppo", "1.1.0")
    mock_manager.delete_checkpoint.assert_any_call("ppo", "1.2.0")


def test_model_retention_dry_run(mock_manager):
    config = RetentionConfig(keep_latest_n=3, dry_run=True)
    policy = ModelRetentionPolicy(config, mock_manager)

    deleted_count = policy.cleanup_versions("ppo")

    assert deleted_count == 3
    # In dry run, actual delete should NOT be called
    assert mock_manager.delete_checkpoint.call_count == 0


def test_model_retention_no_cleanup_needed(mock_manager):
    config = RetentionConfig(keep_latest_n=10)  # More than 6
    policy = ModelRetentionPolicy(config, mock_manager)

    deleted_count = policy.cleanup_versions("ppo")

    assert deleted_count == 0
    assert mock_manager.delete_checkpoint.call_count == 0


def test_enforce_all(mock_manager):
    config = RetentionConfig(keep_latest_n=3)
    policy = ModelRetentionPolicy(config, mock_manager)

    results = policy.enforce_all(["ppo", "vae"])

    assert "ppo" in results
    assert "vae" in results
    assert results["ppo"] == 3
    assert results["vae"] == 3
