"""
Google Cloud Storage backend for models.
"""

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from python.src.storage.base import ModelMetadata, ModelStorage, StorageConfig

if TYPE_CHECKING:
    pass


class GCSStorage(ModelStorage):
    """Google Cloud Storage backend."""

    def __init__(self, config: StorageConfig):
        """Initialize Google Cloud Storage backend."""
        super().__init__(config)
        self._client = None
        self._bucket_name = config.gcs_bucket
        self._prefix = config.gcs_prefix.rstrip("/") + "/"

    @property
    def client(
        self,
    ) -> (
        Any
    ):  # Use Any to avoid mandatory google-cloud-storage dependency for type checking
        """Lazy initialization of GCS client."""
        if self._client is None:
            try:
                # Keep your existing deferred import for runtime
                from google.cloud import storage

                if self.config.gcs_credentials_path:
                    self._client = storage.Client.from_service_account_json(
                        self.config.gcs_credentials_path
                    )
                else:
                    self._client = storage.Client()
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required for GCS storage. "
                    "Install it with: pip install google-cloud-storage"
                ) from None
        return self._client

    @property
    def bucket(self) -> Any:
        """Get the GCS bucket object."""
        return self.client.bucket(self._bucket_name)

    def _model_path(self, name: str, version: str) -> str:
        """Get GCS path for a model version."""
        return f"{self._prefix}{name}/{version}.pt"

    def _metadata_path(self, name: str, version: str) -> str:
        """Get GCS path for metadata."""
        return f"{self._prefix}{name}/{version}.json"

    def _latest_path(self, name: str) -> str:
        """Get GCS path for latest version pointer."""
        return f"{self._prefix}{name}/latest"

    def _generate_version(self) -> str:
        """Generate a new version string."""
        return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    def save(
        self,
        model_data: bytes,
        name: str,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save model to GCS."""
        if version is None:
            version = self._generate_version()

        # Compress data
        compressed_data = self._compress(model_data)

        # Upload model
        model_blob = self.bucket.blob(self._model_path(name, version))
        model_blob.metadata = {
            "model-name": name,
            "model-version": version,
            "checksum": self.compute_checksum(model_data),
        }
        model_blob.upload_from_string(
            compressed_data,
            content_type="application/octet-stream",
        )

        # Upload metadata
        meta = ModelMetadata(
            name=name,
            version=version,
            checksum=self.compute_checksum(model_data),
            size_bytes=len(model_data),
            created_at=datetime.now(UTC).isoformat(),
            **(metadata or {}),
        )
        meta_blob = self.bucket.blob(self._metadata_path(name, version))
        meta_blob.upload_from_string(
            json.dumps(meta.__dict__, indent=2),
            content_type="application/json",
        )

        # Update latest pointer
        latest_blob = self.bucket.blob(self._latest_path(name))
        latest_blob.upload_from_string(
            version,
            content_type="text/plain",
        )

        # Cache locally
        if self.config.cache_enabled:
            self._save_to_cache(model_data, name, version)

        # Clean up old versions
        if self.config.versioning:
            self._cleanup_old_versions(name)

        return f"gs://{self._bucket_name}/{self._model_path(name, version)}"

    def load(self, name: str, version: str | None = None) -> bytes:
        """Load model from GCS."""
        if version is None:
            version = self._get_latest_version(name)
            if version is None:
                raise FileNotFoundError(f"Model '{name}' not found in GCS")

        # Check cache first
        if self.config.cache_enabled:
            cached = self._load_from_cache(name, version)
            if cached is not None:
                return cached

        # Download from GCS
        blob = self.bucket.blob(self._model_path(name, version))
        if not blob.exists():
            raise FileNotFoundError(
                f"Model '{name}' version '{version}' not found in GCS"
            )

        compressed_data = blob.download_as_bytes()
        data = self._decompress(compressed_data)

        # Cache locally
        if self.config.cache_enabled:
            self._save_to_cache(data, name, version)

        return data

    def exists(self, name: str, version: str | None = None) -> bool:
        """Check if model exists in GCS."""
        if version is None:
            # Check if any version exists
            prefix = f"{self._prefix}{name}/"
            blobs = list(self.bucket.list_blobs(prefix=prefix, max_results=1))
            return len(blobs) > 0
        else:
            blob = self.bucket.blob(self._model_path(name, version))
            return bool(blob.exists())

    def delete(self, name: str, version: str | None = None) -> bool:
        """Delete model from GCS."""
        try:
            if version is None:
                # Delete all versions
                prefix = f"{self._prefix}{name}/"
                blobs = self.bucket.list_blobs(prefix=prefix)
                for blob in blobs:
                    blob.delete()
                return True
            else:
                # Delete specific version
                model_blob = self.bucket.blob(self._model_path(name, version))
                meta_blob = self.bucket.blob(self._metadata_path(name, version))
                if model_blob.exists():
                    model_blob.delete()
                if meta_blob.exists():
                    meta_blob.delete()
                return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List all model names in GCS."""
        # List "directories" under the prefix
        iterator = self.client.list_blobs(
            self._bucket_name,
            prefix=self._prefix,
            delimiter="/",
        )
        # Consume the iterator to get prefixes
        list(iterator)  # Must consume to populate prefixes
        models = []
        for prefix in iterator.prefixes:
            name = prefix.rstrip("/").split("/")[-1]
            if name:
                models.append(name)
        return models

    def list_versions(self, name: str) -> list[str]:
        """List all versions of a model in GCS."""
        prefix = f"{self._prefix}{name}/"
        blobs = self.bucket.list_blobs(prefix=prefix)
        versions = []
        for blob in blobs:
            if blob.name.endswith(".pt"):
                version = blob.name.split("/")[-1].replace(".pt", "")
                versions.append(version)
        return sorted(versions, reverse=True)

    def get_metadata(self, name: str, version: str | None = None) -> ModelMetadata:
        """Get metadata for a model from GCS."""
        if version is None:
            version = self._get_latest_version(name)
            if version is None:
                raise FileNotFoundError(f"Model '{name}' not found in GCS")

        blob = self.bucket.blob(self._metadata_path(name, version))
        if not blob.exists():
            raise FileNotFoundError(
                f"Metadata for '{name}' version '{version}' not found in GCS"
            )

        data = json.loads(blob.download_as_text())
        return ModelMetadata(**data)

    def _get_latest_version(self, name: str) -> str | None:
        """Get the latest version string for a model."""
        blob = self.bucket.blob(self._latest_path(name))
        if blob.exists():
            return cast(str, blob.download_as_text().strip())

        # Fallback: find most recent version
        versions = self.list_versions(name)
        return versions[0] if versions else None

    def _cleanup_old_versions(self, name: str) -> None:
        """Remove old versions beyond max_versions."""
        versions = self.list_versions(name)
        if len(versions) > self.config.max_versions:
            for old_version in versions[self.config.max_versions :]:
                self.delete(name, old_version)
