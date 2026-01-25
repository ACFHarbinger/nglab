import sys
from unittest.mock import MagicMock, patch

import pytest

from python.src.storage.base import StorageConfig
from python.src.storage.gcs import GCSStorage


@pytest.fixture
def mock_gcs_lib():
    mock_storage = MagicMock()
    mock_cloud = MagicMock()
    mock_cloud.storage = mock_storage

    with patch.dict(
        sys.modules,
        {
            "google": MagicMock(),
            "google.cloud": mock_cloud,
            "google.cloud.storage": mock_storage,
        },
    ):
        yield mock_storage


@pytest.fixture
def gcs_storage(mock_gcs_lib):
    config = StorageConfig(
        storage_type="gcs",
        gcs_bucket="test-bucket",
        gcs_prefix="models",
        compression="none",
    )
    return GCSStorage(config)


class TestGCSStorage:
    def test_client_initialization(self, gcs_storage, mock_gcs_lib):
        # Trigger lazy init
        client = gcs_storage.client
        assert client is not None
        mock_gcs_lib.Client.assert_called()

    def test_save_model(self, gcs_storage, mock_gcs_lib):
        # Mock bucket and blob
        client_mock = mock_gcs_lib.Client.return_value
        bucket_mock = client_mock.bucket.return_value
        blob_mock = bucket_mock.blob.return_value

        path = gcs_storage.save(b"data", "model", "v1")

        assert path == "gs://test-bucket/models/model/v1.pt"
        assert blob_mock.upload_from_string.called

    def test_load_model(self, gcs_storage, mock_gcs_lib):
        client_mock = mock_gcs_lib.Client.return_value
        bucket_mock = client_mock.bucket.return_value
        blob_mock = bucket_mock.blob.return_value

        blob_mock.exists.return_value = True
        blob_mock.download_as_bytes.return_value = b"data"

        data = gcs_storage.load("model", "v1")
        assert data == b"data"

    def test_exists(self, gcs_storage, mock_gcs_lib):
        client_mock = mock_gcs_lib.Client.return_value
        bucket_mock = client_mock.bucket.return_value
        blob_mock = bucket_mock.blob.return_value

        blob_mock.exists.return_value = True
        assert gcs_storage.exists("model", "v1") is True

        blob_mock.exists.return_value = False
        assert gcs_storage.exists("model", "v1") is False

    def test_list_models(self, gcs_storage, mock_gcs_lib):
        client_mock = mock_gcs_lib.Client.return_value
        # mock list_blobs iterator
        iterator_mock = MagicMock()
        iterator_mock.prefixes = set(["models/m1/", "models/m2/"])
        iterator_mock.__iter__.return_value = (
            []
        )  # Yields nothing (blobs), only care about prefixes

        client_mock.list_blobs.return_value = iterator_mock

        models = gcs_storage.list_models()
        assert sorted(models) == ["m1", "m2"]
