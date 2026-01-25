import sys
from unittest.mock import ANY, MagicMock, patch

import pytest

from python.src.storage.base import StorageConfig

# Set PYTHONPATH via pytest or env is assumed.
from python.src.storage.s3 import S3Storage


@pytest.fixture
def mock_boto3():
    mock = MagicMock()
    with patch.dict(
        sys.modules,
        {"boto3": mock, "botocore": MagicMock(), "botocore.config": MagicMock()},
    ):
        yield mock


@pytest.fixture
def s3_storage(mock_boto3, tmp_path):
    config = StorageConfig(
        storage_type="s3",
        s3_bucket="test-bucket",
        s3_prefix="models",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        cache_dir=str(tmp_path / "cache"),
    )
    return S3Storage(config)


class TestS3Storage:
    def test_client_initialization(self, s3_storage, mock_boto3):
        client = s3_storage.client
        assert client is not None
        mock_boto3.client.assert_called_with(
            "s3",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            config=ANY,  # botocore config
        )

    def test_save_model(self, s3_storage, mock_boto3):
        client_mock = mock_boto3.client.return_value
        model_data = b"model_data"
        name = "test_model"
        version = "v1"

        path = s3_storage.save(model_data, name, version=version)

        assert path == "s3://test-bucket/models/test_model/v1.pt"

        # Verify put_object calls
        # 1. Model
        # 2. Metadata
        # 3. Latest pointer
        assert client_mock.put_object.call_count == 3

        # Check model upload
        client_mock.put_object.assert_any_call(
            Bucket="test-bucket",
            Key="models/test_model/v1.pt",
            Body=ANY,  # Compressed data
            ContentType="application/octet-stream",
            Metadata=ANY,
        )

    def test_load_model(self, s3_storage, mock_boto3):
        client_mock = mock_boto3.client.return_value

        # Mock response body
        body_mock = MagicMock()
        body_mock.read.return_value = (
            b"compressed_data"  # Assuming compression logic handles this or dummy
        )

        # Need s3_storage._decompress to just return data if we want to simplify,
        # but let's assume default config uses zstd or gzip.
        # If we set compression="none", it's easier.
        s3_storage.config.compression = "none"
        body_mock.read.return_value = b"model_data"

        client_mock.get_object.return_value = {"Body": body_mock}

        data = s3_storage.load("test_model", "v1")
        assert data == b"model_data"

        client_mock.get_object.assert_called_with(
            Bucket="test-bucket", Key="models/test_model/v1.pt"
        )

    def test_exists(self, s3_storage, mock_boto3):
        client_mock = mock_boto3.client.return_value

        exists = s3_storage.exists("test_model", "v1")
        assert exists is True
        client_mock.head_object.assert_called_with(
            Bucket="test-bucket", Key="models/test_model/v1.pt"
        )

        # Test not exists exception
        client_mock.head_object.side_effect = Exception("Not Found")
        assert s3_storage.exists("test_model", "v1") is False

    def test_list_models(self, s3_storage, mock_boto3):
        client_mock = mock_boto3.client.return_value
        client_mock.list_objects_v2.return_value = {
            "CommonPrefixes": [{"Prefix": "models/m1/"}, {"Prefix": "models/m2/"}]
        }

        models = s3_storage.list_models()
        assert models == ["m1", "m2"]

    def test_delete_version(self, s3_storage, mock_boto3):
        client_mock = mock_boto3.client.return_value

        s3_storage.delete("test_model", "v1")

        # Should delete model and metadata
        assert client_mock.delete_object.call_count == 2
