"""
Base classes for model storage backends.

Defines the abstract interface that all storage backends must implement.
"""

import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast


@dataclass
class StorageConfig:
    """Configuration for model storage backends."""

    # Storage type: local, s3, gcs
    storage_type: str = "local"

    # Local storage settings
    local_path: str = "./model_weights"

    # S3 settings
    s3_bucket: str = ""
    s3_prefix: str = "models/"
    s3_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # GCS settings
    gcs_bucket: str = ""
    gcs_prefix: str = "models/"
    gcs_credentials_path: str | None = None

    # Common settings
    compression: str = "zstd"  # none, gzip, zstd
    versioning: bool = True
    max_versions: int = 5

    # Cache settings
    cache_enabled: bool = True
    cache_dir: str = ".cache/models"
    cache_max_size_gb: float = 10.0

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Create config from environment variables."""
        return cls(
            storage_type=os.getenv("NGLAB_MODEL_STORAGE", "local"),
            local_path=os.getenv("MODEL_WEIGHTS_DIR", "./model_weights"),
            s3_bucket=os.getenv("S3_BUCKET_NAME", ""),
            s3_prefix=os.getenv("S3_PREFIX", "models/"),
            s3_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            gcs_bucket=os.getenv("GCS_BUCKET_NAME", ""),
            gcs_prefix=os.getenv("GCS_PREFIX", "models/"),
            gcs_credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )


@dataclass
class ModelMetadata:
    """Metadata for stored models."""

    name: str
    version: str
    checksum: str
    size_bytes: int
    created_at: str
    framework: str = "pytorch"
    architecture: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class ModelStorage(ABC):
    """Abstract base class for model storage backends."""

    def __init__(self, config: StorageConfig) -> None:
        """Initialize the storage backend."""
        self.config = config
        self._setup_cache()

    def _setup_cache(self) -> None:
        """Setup local cache directory."""
        if self.config.cache_enabled:
            cache_path = Path(self.config.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def save(
        self,
        model_data: bytes,
        name: str,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save model data to storage.

        Args:
            model_data: Serialized model bytes
            name: Model name
            version: Optional version string (auto-generated if not provided)
            metadata: Optional metadata dictionary

        Returns:
            Full path/URI to saved model
        """
        pass

    @abstractmethod
    def load(
        self,
        name: str,
        version: str | None = None,
    ) -> bytes:
        """
        Load model data from storage.

        Args:
            name: Model name
            version: Optional version (latest if not provided)

        Returns:
            Serialized model bytes
        """
        pass

    @abstractmethod
    def exists(self, name: str, version: str | None = None) -> bool:
        """Check if a model exists in storage."""
        pass

    @abstractmethod
    def delete(self, name: str, version: str | None = None) -> bool:
        """Delete a model from storage."""
        pass

    @abstractmethod
    def list_models(self) -> list[str]:
        """List all model names in storage."""
        pass

    @abstractmethod
    def list_versions(self, name: str) -> list[str]:
        """List all versions of a model."""
        pass

    @abstractmethod
    def get_metadata(self, name: str, version: str | None = None) -> ModelMetadata:
        """Get metadata for a model."""
        pass

    def save_file(
        self,
        file_path: Path,
        name: str,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save model from file path."""
        with open(file_path, "rb") as f:
            return self.save(f.read(), name, version, metadata)

    def load_to_file(
        self,
        name: str,
        output_path: Path,
        version: str | None = None,
    ) -> Path:
        """Load model to a file path."""
        data = self.load(name, version)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(data)
        return output_path

    @staticmethod
    def compute_checksum(data: bytes) -> str:
        """Compute SHA256 checksum of data."""
        return hashlib.sha256(data).hexdigest()

    def _get_cache_path(self, name: str, version: str) -> Path:
        """Get local cache path for a model."""
        return Path(self.config.cache_dir) / name / f"{version}.pt"

    def _save_to_cache(self, data: bytes, name: str, version: str) -> Path:
        """Save model data to local cache."""
        cache_path = self._get_cache_path(name, version)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(data)
        return cache_path

    def _load_from_cache(self, name: str, version: str) -> bytes | None:
        """Load model data from local cache if available."""
        cache_path = self._get_cache_path(name, version)
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return f.read()
        return None

    def _compress(self, data: bytes) -> bytes:
        """Compress data based on config."""
        if self.config.compression == "none":
            return data
        elif self.config.compression == "gzip":
            import gzip

            return gzip.compress(data, compresslevel=6)
        elif self.config.compression == "zstd":
            try:
                import zstandard as zstd

                cctx = zstd.ZstdCompressor(level=3)
                return cast(bytes, cctx.compress(data))
            except ImportError:
                # Fallback to gzip if zstd not available
                import gzip

                return gzip.compress(data, compresslevel=6)
        return data

    def _decompress(self, data: bytes) -> bytes:
        """Decompress data based on config."""
        if self.config.compression == "none":
            return data
        elif self.config.compression == "gzip":
            import gzip

            return gzip.decompress(data)
        elif self.config.compression == "zstd":
            try:
                import zstandard as zstd

                dctx = zstd.ZstdDecompressor()
                return cast(bytes, dctx.decompress(data))
            except ImportError:
                import gzip

                return gzip.decompress(data)

        return data
