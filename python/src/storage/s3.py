"""
AWS S3 storage backend for models.
"""

import json
from datetime import UTC, datetime
from typing import Any, cast

from python.src.storage.base import ModelMetadata, ModelStorage, StorageConfig


class S3Storage(ModelStorage):
    """AWS S3 storage backend."""

    def __init__(self, config: StorageConfig):
        """Initialize AWS S3 storage backend."""
        super().__init__(config)
        self._client = None
        self._bucket = config.s3_bucket
        self._prefix = config.s3_prefix.rstrip("/") + "/"

    @property
    def client(self) -> Any:
        """Lazy initialization of S3 client."""
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config

                boto_config = Config(
                    region_name=self.config.s3_region,
                    retries={"max_attempts": 3, "mode": "adaptive"},
                )

                if self.config.aws_access_key_id and self.config.aws_secret_access_key:
                    self._client = boto3.client(
                        "s3",
                        aws_access_key_id=self.config.aws_access_key_id,
                        aws_secret_access_key=self.config.aws_secret_access_key,
                        config=boto_config,
                    )
                else:
                    # Use default credentials (IAM role, env vars, etc.)
                    self._client = boto3.client("s3", config=boto_config)
            except ImportError:
                raise ImportError(
                    "boto3 is required for S3 storage. "
                    "Install it with: pip install boto3"
                ) from None
        return self._client

    def _model_key(self, name: str, version: str) -> str:
        """Get S3 key for a model version."""
        return f"{self._prefix}{name}/{version}.pt"

    def _metadata_key(self, name: str, version: str) -> str:
        """Get S3 key for metadata."""
        return f"{self._prefix}{name}/{version}.json"

    def _latest_key(self, name: str) -> str:
        """Get S3 key for latest version pointer."""
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
        """Save model to S3."""
        if version is None:
            version = self._generate_version()

        # Compress data
        compressed_data = self._compress(model_data)

        # Upload model
        model_key = self._model_key(name, version)
        self.client.put_object(
            Bucket=self._bucket,
            Key=model_key,
            Body=compressed_data,
            ContentType="application/octet-stream",
            Metadata={
                "model-name": name,
                "model-version": version,
                "checksum": self.compute_checksum(model_data),
            },
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
        self.client.put_object(
            Bucket=self._bucket,
            Key=self._metadata_key(name, version),
            Body=json.dumps(meta.__dict__, indent=2),
            ContentType="application/json",
        )

        # Update latest pointer
        self.client.put_object(
            Bucket=self._bucket,
            Key=self._latest_key(name),
            Body=version.encode("utf-8"),
            ContentType="text/plain",
        )

        # Also cache locally
        if self.config.cache_enabled:
            self._save_to_cache(model_data, name, version)

        # Clean up old versions
        if self.config.versioning:
            self._cleanup_old_versions(name)

        return f"s3://{self._bucket}/{model_key}"

    def load(self, name: str, version: str | None = None) -> bytes:
        """Load model from S3."""
        if version is None:
            version = self._get_latest_version(name)
            if version is None:
                raise FileNotFoundError(f"Model '{name}' not found in S3")

        # Check cache first
        if self.config.cache_enabled:
            cached = self._load_from_cache(name, version)
            if cached is not None:
                return cached

        # Download from S3
        model_key = self._model_key(name, version)
        try:
            response = self.client.get_object(Bucket=self._bucket, Key=model_key)
            compressed_data = response["Body"].read()
        except self.client.exceptions.NoSuchKey:
            raise FileNotFoundError(
                f"Model '{name}' version '{version}' not found in S3"
            ) from None

        data = self._decompress(compressed_data)

        # Cache locally
        if self.config.cache_enabled:
            self._save_to_cache(data, name, version)

        return data

    def exists(self, name: str, version: str | None = None) -> bool:
        """Check if model exists in S3."""
        try:
            if version is None:
                # Check if any version exists
                response = self.client.list_objects_v2(
                    Bucket=self._bucket,
                    Prefix=f"{self._prefix}{name}/",
                    MaxKeys=1,
                )
                return bool(response.get("KeyCount", 0) > 0)
            else:
                self.client.head_object(
                    Bucket=self._bucket,
                    Key=self._model_key(name, version),
                )
                return True
        except Exception:
            return False

    def delete(self, name: str, version: str | None = None) -> bool:
        """Delete model from S3."""
        try:
            if version is None:
                # Delete all versions
                response = self.client.list_objects_v2(
                    Bucket=self._bucket,
                    Prefix=f"{self._prefix}{name}/",
                )
                if "Contents" in response:
                    objects = [{"Key": obj["Key"]} for obj in response["Contents"]]
                    self.client.delete_objects(
                        Bucket=self._bucket,
                        Delete={"Objects": objects},
                    )
                return True
            else:
                # Delete specific version
                self.client.delete_object(
                    Bucket=self._bucket,
                    Key=self._model_key(name, version),
                )
                self.client.delete_object(
                    Bucket=self._bucket,
                    Key=self._metadata_key(name, version),
                )
                return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List all model names in S3."""
        response = self.client.list_objects_v2(
            Bucket=self._bucket,
            Prefix=self._prefix,
            Delimiter="/",
        )
        models = []
        for prefix in response.get("CommonPrefixes", []):
            # Extract model name from prefix
            name = prefix["Prefix"].rstrip("/").split("/")[-1]
            if name:
                models.append(name)
        return models

    def list_versions(self, name: str) -> list[str]:
        """List all versions of a model in S3."""
        response = self.client.list_objects_v2(
            Bucket=self._bucket,
            Prefix=f"{self._prefix}{name}/",
        )
        versions = []
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".pt"):
                version = key.split("/")[-1].replace(".pt", "")
                versions.append(version)
        return sorted(versions, reverse=True)

    def get_metadata(self, name: str, version: str | None = None) -> ModelMetadata:
        """Get metadata for a model from S3."""
        if version is None:
            version = self._get_latest_version(name)
            if version is None:
                raise FileNotFoundError(f"Model '{name}' not found in S3")

        try:
            response = self.client.get_object(
                Bucket=self._bucket,
                Key=self._metadata_key(name, version),
            )
            data = json.loads(response["Body"].read().decode("utf-8"))
            return ModelMetadata(**data)
        except Exception as e:
            raise FileNotFoundError(
                f"Metadata for '{name}' version '{version}' not found: {e}"
            ) from e

    def _get_latest_version(self, name: str) -> str | None:
        """Get the latest version string for a model."""
        try:
            response = self.client.get_object(
                Bucket=self._bucket,
                Key=self._latest_key(name),
            )
            return cast(str, response["Body"].read().decode("utf-8").strip())
        except Exception:
            # Fallback: find most recent version
            versions = self.list_versions(name)
            return versions[0] if versions else None

    def _cleanup_old_versions(self, name: str) -> None:
        """Remove old versions beyond max_versions."""
        versions = self.list_versions(name)
        if len(versions) > self.config.max_versions:
            for old_version in versions[self.config.max_versions :]:
                self.delete(name, old_version)
