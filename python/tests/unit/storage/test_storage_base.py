"""
Tests for the storage base module - compression, caching, checksum utilities.
"""

import gzip
from unittest.mock import patch

import pytest

from python.src.storage.base import ModelMetadata, ModelStorage, StorageConfig


class ConcreteStorage(ModelStorage):
    """Concrete implementation for testing abstract base class methods."""

    def save(self, model_data, name, version=None, metadata=None):
        return f"saved/{name}/{version}"

    def load(self, name, version=None):
        return b"loaded_data"

    def exists(self, name, version=None):
        return True

    def delete(self, name, version=None):
        return True

    def list_models(self):
        return ["model1", "model2"]

    def list_versions(self, name):
        return ["v1", "v2"]

    def get_metadata(self, name, version=None):
        return ModelMetadata(
            name=name,
            version=version or "v1",
            checksum="abc123",
            size_bytes=100,
            created_at="2024-01-01T00:00:00",
        )


class TestStorageConfig:
    def test_default_values(self):
        config = StorageConfig()
        assert config.storage_type == "local"
        assert config.local_path == "./model_weights"
        assert config.compression == "zstd"
        assert config.versioning is True
        assert config.max_versions == 5
        assert config.cache_enabled is True

    def test_custom_values(self):
        config = StorageConfig(
            storage_type="s3",
            s3_bucket="my-bucket",
            s3_prefix="models/prod/",
            compression="gzip",
            max_versions=10,
        )
        assert config.storage_type == "s3"
        assert config.s3_bucket == "my-bucket"
        assert config.s3_prefix == "models/prod/"
        assert config.compression == "gzip"
        assert config.max_versions == 10

    def test_from_env(self):
        with patch.dict(
            "os.environ",
            {
                "NGLAB_MODEL_STORAGE": "s3",
                "MODEL_WEIGHTS_DIR": "/custom/path",
                "S3_BUCKET_NAME": "test-bucket",
                "AWS_DEFAULT_REGION": "eu-west-1",
            },
        ):
            config = StorageConfig.from_env()
            assert config.storage_type == "s3"
            assert config.local_path == "/custom/path"
            assert config.s3_bucket == "test-bucket"
            assert config.s3_region == "eu-west-1"

    def test_from_env_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            config = StorageConfig.from_env()
            assert config.storage_type == "local"
            assert config.local_path == "./model_weights"


class TestModelMetadata:
    def test_required_fields(self):
        meta = ModelMetadata(
            name="test_model",
            version="v1.0",
            checksum="sha256hash",
            size_bytes=1024,
            created_at="2024-01-01T00:00:00Z",
        )
        assert meta.name == "test_model"
        assert meta.version == "v1.0"
        assert meta.checksum == "sha256hash"
        assert meta.size_bytes == 1024

    def test_optional_fields(self):
        meta = ModelMetadata(
            name="test",
            version="v1",
            checksum="abc",
            size_bytes=100,
            created_at="2024-01-01",
            framework="tensorflow",
            architecture="transformer",
            metrics={"accuracy": 0.95},
            tags=["production", "v1"],
            extra={"custom_key": "custom_value"},
        )
        assert meta.framework == "tensorflow"
        assert meta.architecture == "transformer"
        assert meta.metrics == {"accuracy": 0.95}
        assert meta.tags == ["production", "v1"]
        assert meta.extra == {"custom_key": "custom_value"}

    def test_default_optional_fields(self):
        meta = ModelMetadata(
            name="test",
            version="v1",
            checksum="abc",
            size_bytes=100,
            created_at="2024-01-01",
        )
        assert meta.framework == "pytorch"
        assert meta.architecture == ""
        assert meta.metrics == {}
        assert meta.tags == []
        assert meta.extra == {}


class TestModelStorageBase:
    @pytest.fixture
    def storage(self, tmp_path):
        config = StorageConfig(
            storage_type="local",
            local_path=str(tmp_path / "models"),
            cache_dir=str(tmp_path / "cache"),
            cache_enabled=True,
        )
        return ConcreteStorage(config)

    def test_setup_cache_creates_directory(self, tmp_path):
        cache_dir = tmp_path / "test_cache"
        config = StorageConfig(
            cache_enabled=True,
            cache_dir=str(cache_dir),
        )
        ConcreteStorage(config)
        assert cache_dir.exists()

    def test_setup_cache_disabled(self, tmp_path):
        cache_dir = tmp_path / "no_cache"
        config = StorageConfig(
            cache_enabled=False,
            cache_dir=str(cache_dir),
        )
        ConcreteStorage(config)
        # Cache directory should not be created when disabled
        assert not cache_dir.exists()

    def test_compute_checksum(self, storage):
        data = b"test data for checksum"
        checksum = storage.compute_checksum(data)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex digest length
        # Same data should produce same checksum
        assert storage.compute_checksum(data) == checksum
        # Different data should produce different checksum
        assert storage.compute_checksum(b"different data") != checksum

    def test_compress_decompress_none(self, tmp_path):
        config = StorageConfig(compression="none", cache_dir=str(tmp_path / "cache"))
        storage = ConcreteStorage(config)
        data = b"uncompressed data"
        compressed = storage._compress(data)
        assert compressed == data
        decompressed = storage._decompress(compressed)
        assert decompressed == data

    def test_compress_decompress_gzip(self, tmp_path):
        config = StorageConfig(compression="gzip", cache_dir=str(tmp_path / "cache"))
        storage = ConcreteStorage(config)
        data = b"data to compress with gzip" * 100
        compressed = storage._compress(data)
        # Compressed data should be smaller (for repetitive data)
        assert len(compressed) < len(data)
        # Should be valid gzip data
        assert gzip.decompress(compressed) == data
        # Round-trip should work
        decompressed = storage._decompress(compressed)
        assert decompressed == data

    def test_compress_decompress_zstd(self):
        storage = ConcreteStorage(StorageConfig())
        data = (
            b"test data for compression" * 100
        )  # Make data larger so compression works

        try:
            import zstandard  # noqa: F401
        except (ImportError, AttributeError):
            pytest.skip("zstandard not available")

        compressed = storage._compress(data)
        assert len(compressed) < len(data)
        decompressed = storage._decompress(compressed)
        assert decompressed == data

    def test_compress_zstd_fallback_to_gzip(self, tmp_path):
        """Test that zstd compression falls back to gzip when zstd is not installed."""
        config = StorageConfig(compression="zstd", cache_dir=str(tmp_path / "cache"))
        storage = ConcreteStorage(config)
        data = b"fallback test data" * 50

        with patch.dict("sys.modules", {"zstandard": None}):
            # This should fall back to gzip
            compressed = storage._compress(data)
            decompressed = storage._decompress(compressed)
            assert decompressed == data

    def test_get_cache_path(self, storage):
        cache_path = storage._get_cache_path("my_model", "v1.0")
        assert "my_model" in str(cache_path)
        assert "v1.0.pt" in str(cache_path)

    def test_save_to_cache(self, storage):
        data = b"model weights data"
        cache_path = storage._save_to_cache(data, "test_model", "v1")
        assert cache_path.exists()
        with open(cache_path, "rb") as f:
            assert f.read() == data

    def test_load_from_cache_exists(self, storage):
        data = b"cached model data"
        storage._save_to_cache(data, "cached_model", "v1")
        loaded = storage._load_from_cache("cached_model", "v1")
        assert loaded == data

    def test_load_from_cache_not_exists(self, storage):
        loaded = storage._load_from_cache("nonexistent", "v1")
        assert loaded is None

    def test_save_file(self, storage, tmp_path):
        # Create a test file
        test_file = tmp_path / "model.bin"
        test_file.write_bytes(b"model file content")

        result = storage.save_file(test_file, "file_model", "v1")
        assert "file_model" in result

    def test_load_to_file(self, storage, tmp_path):
        output_path = tmp_path / "output" / "loaded_model.bin"
        result = storage.load_to_file("any_model", output_path, "v1")
        assert result == output_path
        assert output_path.exists()
        assert output_path.read_bytes() == b"loaded_data"


class TestCompressionEdgeCases:
    @pytest.fixture
    def storage_no_compression(self, tmp_path):
        config = StorageConfig(
            compression="none",
            cache_dir=str(tmp_path / "cache"),
        )
        return ConcreteStorage(config)

    def test_empty_data_compression(self, tmp_path):
        config = StorageConfig(compression="gzip", cache_dir=str(tmp_path / "cache"))
        storage = ConcreteStorage(config)
        data = b""
        compressed = storage._compress(data)
        decompressed = storage._decompress(compressed)
        assert decompressed == data

    def test_large_data_compression(self, tmp_path):
        config = StorageConfig(compression="gzip", cache_dir=str(tmp_path / "cache"))
        storage = ConcreteStorage(config)
        # 1MB of data
        data = b"x" * (1024 * 1024)
        compressed = storage._compress(data)
        assert len(compressed) < len(data)
        decompressed = storage._decompress(compressed)
        assert decompressed == data

    def test_binary_data_compression(self, tmp_path):
        config = StorageConfig(compression="gzip", cache_dir=str(tmp_path / "cache"))
        storage = ConcreteStorage(config)
        # Random binary data
        import os

        data = os.urandom(1000)
        compressed = storage._compress(data)
        decompressed = storage._decompress(compressed)
        assert decompressed == data

    def test_unknown_compression_passthrough(self, tmp_path):
        """Test that unknown compression type returns data unchanged."""
        config = StorageConfig(compression="unknown", cache_dir=str(tmp_path / "cache"))
        storage = ConcreteStorage(config)
        data = b"test data"
        compressed = storage._compress(data)
        assert compressed == data
        decompressed = storage._decompress(compressed)
        assert decompressed == data
