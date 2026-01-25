import os
from unittest.mock import patch

import pytest

from python.src.storage.base import StorageConfig
from python.src.storage.factory import create_storage
from python.src.storage.gcs import GCSStorage
from python.src.storage.local import LocalStorage
from python.src.storage.s3 import S3Storage


class TestStorageFactory:
    def test_create_local_storage(self, tmp_path):
        config = StorageConfig(storage_type="local", local_path=str(tmp_path))
        storage = create_storage(config)
        assert isinstance(storage, LocalStorage)

    def test_create_s3_storage(self):
        config = StorageConfig(storage_type="s3", s3_bucket="my-bucket")
        storage = create_storage(config)
        assert isinstance(storage, S3Storage)

    def test_create_gcs_storage(self):
        config = StorageConfig(storage_type="gcs", gcs_bucket="my-bucket")
        storage = create_storage(config)
        assert isinstance(storage, GCSStorage)

    def test_create_from_env(self):
        with patch.dict(
            os.environ,
            {"NGLAB_MODEL_STORAGE": "local", "MODEL_WEIGHTS_DIR": "/tmp/test"},
        ):
            storage = create_storage()
            assert isinstance(storage, LocalStorage)
            assert str(storage.base_path) == "/tmp/test"

    def test_unknown_storage_type(self):
        config = StorageConfig(storage_type="ftp")
        with pytest.raises(ValueError, match="Unknown storage type"):
            create_storage(config)

    def test_s3_missing_bucket(self):
        config = StorageConfig(storage_type="s3", s3_bucket=None)
        with pytest.raises(ValueError, match="S3 bucket name is required"):
            create_storage(config)

    def test_gcs_missing_bucket(self):
        config = StorageConfig(storage_type="gcs", gcs_bucket=None)
        with pytest.raises(ValueError, match="GCS bucket name is required"):
            create_storage(config)
