import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from python.src.utils.io.cloud_storage import (
    CloudCheckpointManager,
    CloudStorageConfig,
    GCSBackend,
    S3Backend,
    create_cloud_manager_from_env,
)


@pytest.fixture
def config():
    return CloudStorageConfig(
        bucket="test-bucket", prefix="test-prefix", fallback_local_path="fallback"
    )


@pytest.fixture
def dummy_model():
    return nn.Linear(10, 2)


@pytest.fixture
def dummy_optimizer(dummy_model):
    return torch.optim.Adam(dummy_model.parameters())


# --- S3 Backend Tests ---


@patch("boto3.Session")
def test_s3_backend_upload(mock_session_cls, config, tmp_path):
    mock_session = mock_session_cls.return_value
    mock_s3 = mock_session.client.return_value

    backend = S3Backend(config)
    local_file = tmp_path / "test.pt"
    local_file.write_bytes(b"dummy data")

    uri = backend.upload(local_file, "model.pt", {"ver": "1.0"})

    assert uri == "s3://test-bucket/test-prefix/model.pt"
    mock_s3.put_object.assert_called_once()
    _args, kwargs = mock_s3.put_object.call_args
    assert kwargs["Bucket"] == "test-bucket"
    assert kwargs["Key"] == "test-prefix/model.pt"
    assert kwargs["Metadata"] == {"ver": "1.0"}


@patch("boto3.Session")
def test_s3_backend_download(mock_session_cls, config, tmp_path):
    mock_session = mock_session_cls.return_value
    mock_s3 = mock_session.client.return_value

    # Mock zstd compressed data: we need to actually compress something because backend decompress it
    import zstandard as zstd

    data = b"original data"
    compressed = zstd.ZstdCompressor().compress(data)

    mock_resp = {"Body": MagicMock()}
    mock_resp["Body"].read.return_value = compressed
    mock_s3.get_object.return_value = mock_resp

    backend = S3Backend(config)
    local_file = tmp_path / "downloaded.pt"

    backend.download("model.pt", local_file)

    assert local_file.exists()
    assert local_file.read_bytes() == data


# --- GCS Backend Tests ---


@patch("google.cloud.storage.Client")
def test_gcs_backend_upload(mock_client_cls, config, tmp_path):
    mock_client = mock_client_cls.return_value
    mock_bucket = mock_client.bucket.return_value
    mock_blob = mock_bucket.blob.return_value

    backend = GCSBackend(config)
    local_file = tmp_path / "test.pt"
    local_file.write_bytes(b"dummy data")

    uri = backend.upload(local_file, "model.pt", {"ver": "1.0"})

    assert uri == "gs://test-bucket/test-prefix/model.pt"
    mock_blob.upload_from_string.assert_called_once()
    assert mock_blob.metadata == {"ver": "1.0"}


@patch("google.cloud.storage.Client")
def test_gcs_backend_download(mock_client_cls, config, tmp_path):
    mock_client = mock_client_cls.return_value
    mock_bucket = mock_client.bucket.return_value
    mock_blob = mock_bucket.blob.return_value

    import zstandard as zstd

    data = b"original data"
    compressed = zstd.ZstdCompressor().compress(data)

    mock_blob.exists.return_value = True
    mock_blob.download_as_bytes.return_value = compressed

    backend = GCSBackend(config)
    local_file = tmp_path / "downloaded.pt"

    backend.download("model.pt", local_file)

    assert local_file.exists()
    assert local_file.read_bytes() == data


# --- Manager Tests ---


def test_checkpoint_manager_save_load(config, dummy_model, dummy_optimizer):
    with patch("python.src.utils.io.cloud_storage.S3Backend") as mock_backend_cls:
        mock_backend = mock_backend_cls.return_value
        mock_backend.upload.return_value = "s3://test/uri"

        manager = CloudCheckpointManager(config, backend="s3")

        uri = manager.save_checkpoint(
            dummy_model,
            "ppo",
            "1.0.0",
            optimizer=dummy_optimizer,
            metrics={"loss": 0.5},
        )

        assert uri == "s3://test/uri"
        mock_backend.upload.assert_called_once()

        # Now mock load
        # We need to mock torch.load because backend.download saves to a temp file
        # and then manager calls torch.load(temp_path).

        # Save state dicts for comparison
        original_sd = dummy_model.state_dict()

        with patch("torch.load") as mock_torch_load:
            mock_torch_load.return_value = {
                "model_state_dict": original_sd,
                "optimizer_state_dict": dummy_optimizer.state_dict(),
                "metrics": {"loss": 0.5},
                "version": "1.0.0",
            }

            checkpoint = manager.load_checkpoint(
                dummy_model, "ppo", "1.0.0", optimizer=dummy_optimizer
            )

            assert checkpoint["version"] == "1.0.0"
            assert checkpoint["metrics"]["loss"] == 0.5
            mock_backend.download.assert_called_once()


def test_list_versions(config):
    with patch("python.src.utils.io.cloud_storage.S3Backend") as mock_backend_cls:
        mock_backend = mock_backend_cls.return_value
        mock_backend.list_objects.return_value = [
            {"key": "test-prefix/ppo/v1.0.0/checkpoint.pt.zst"},
            {"key": "test-prefix/ppo/v1.1.0/checkpoint.pt.zst"},
        ]

        manager = CloudCheckpointManager(config, backend="s3")
        versions = manager.list_versions("ppo")

        assert versions == ["1.0.0", "1.1.0"]


def test_create_manager_from_env():
    with patch.dict(os.environ, {"NGLAB_CLOUD_BUCKET": "env-bucket"}):
        manager = create_cloud_manager_from_env(backend="s3")
        assert manager.config.bucket == "env-bucket"
        assert isinstance(manager._backend, S3Backend)


@patch("boto3.Session")
def test_s3_backend_fallback(mock_session_cls, config, tmp_path):
    """Test S3 backend falls back to local storage on upload failure."""
    mock_session = mock_session_cls.return_value
    mock_s3 = mock_session.client.return_value
    mock_s3.put_object.side_effect = Exception("S3 failed")

    backend = S3Backend(config)
    local_file = tmp_path / "test.pt"
    local_file.write_bytes(b"data")

    # Should NOT raise, instead use fallback
    uri = backend.upload(local_file, "model.pt", {})

    assert "fallback/model.pt" in uri

    # Cleanup fallback directory
    import shutil

    fallback_dir = Path("fallback")
    if fallback_dir.exists():
        shutil.rmtree(fallback_dir)
