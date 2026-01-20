"""Cloud Storage Backend for Model Checkpoints.

Provides S3 and GCS backends for storing model checkpoints with
automatic compression, versioning, and fallback support.
"""

import logging
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import zstandard as zstd

logger = logging.getLogger(__name__)


@dataclass
class CloudStorageConfig:
    """Configuration for cloud storage backends.

    Attributes:
        bucket: S3 bucket or GCS bucket name.
        prefix: Prefix/folder for models within bucket.
        compression_level: Zstd compression level (1-22, default 3).
        enable_versioning: Enable object versioning.
        fallback_local_path: Local path to use if cloud fails.
    """

    bucket: str
    prefix: str = "models"
    compression_level: int = 3
    enable_versioning: bool = True
    fallback_local_path: str | None = None

    # AWS specific
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # GCS specific
    gcs_project: str | None = None
    gcs_credentials_path: str | None = None


class CloudStorageBackend(ABC):
    """Abstract base class for cloud storage backends."""

    @abstractmethod
    def upload(
        self,
        local_path: Path,
        remote_key: str,
        metadata: dict[str, str],
    ) -> str:
        """Upload a file to cloud storage.

        Args:
            local_path: Local file path to upload.
            remote_key: Remote object key/path.
            metadata: Metadata to attach to object.

        Returns:
            Remote URI of uploaded object.
        """
        ...

    @abstractmethod
    def download(self, remote_key: str, local_path: Path) -> None:
        """Download a file from cloud storage.

        Args:
            remote_key: Remote object key/path.
            local_path: Local path to save file.
        """
        ...

    @abstractmethod
    def list_objects(self, prefix: str) -> list[dict[str, Any]]:
        """List objects with given prefix.

        Args:
            prefix: Prefix to filter objects.

        Returns:
            List of object metadata dictionaries.
        """
        ...

    @abstractmethod
    def delete(self, remote_key: str) -> bool:
        """Delete an object.

        Args:
            remote_key: Remote object key to delete.

        Returns:
            True if deleted, False otherwise.
        """
        ...

    @abstractmethod
    def exists(self, remote_key: str) -> bool:
        """Check if object exists.

        Args:
            remote_key: Remote object key.

        Returns:
            True if exists.
        """
        ...


class S3Backend(CloudStorageBackend):
    """AWS S3 storage backend for model checkpoints.

    Uses boto3 for S3 operations with optional IAM role assumption.
    """

    def __init__(self, config: CloudStorageConfig):
        """Initialize S3 backend.

        Args:
            config: Cloud storage configuration.
        """
        self.config = config
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        """Lazy initialization of boto3 S3 client."""
        if self._client is None:
            try:
                import boto3

                session_kwargs: dict[str, Any] = {
                    "region_name": self.config.aws_region,
                }

                if self.config.aws_access_key_id:
                    session_kwargs["aws_access_key_id"] = self.config.aws_access_key_id
                    session_kwargs["aws_secret_access_key"] = (
                        self.config.aws_secret_access_key
                    )

                session = boto3.Session(**session_kwargs)
                self._client = session.client("s3")
                logger.info(f"S3 client initialized for bucket: {self.config.bucket}")
            except ImportError:
                raise ImportError(
                    "boto3 is required for S3 backend. Install with: pip install boto3"
                )
        return self._client

    def _get_full_key(self, remote_key: str) -> str:
        """Get full S3 key including prefix."""
        return f"{self.config.prefix}/{remote_key}".strip("/")

    def upload(
        self,
        local_path: Path,
        remote_key: str,
        metadata: dict[str, str],
    ) -> str:
        """Upload file to S3 with optional compression."""
        full_key = self._get_full_key(remote_key)

        try:
            with open(local_path, "rb") as f:
                data = f.read()

            # Apply zstd compression
            compressor = zstd.ZstdCompressor(level=self.config.compression_level)
            compressed = compressor.compress(data)

            self.client.put_object(
                Bucket=self.config.bucket,
                Key=full_key,
                Body=compressed,
                Metadata=metadata,
                ContentEncoding="zstd",
            )

            uri = f"s3://{self.config.bucket}/{full_key}"
            logger.info(f"Uploaded checkpoint to {uri}")
            return uri

        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            if self.config.fallback_local_path:
                return self._fallback_save(local_path, remote_key)
            raise

    def _fallback_save(self, local_path: Path, remote_key: str) -> str:
        """Save to local fallback path on cloud failure."""
        fallback = Path(self.config.fallback_local_path or ".")
        fallback_path = fallback / remote_key
        fallback_path.parent.mkdir(parents=True, exist_ok=True)

        import shutil

        shutil.copy2(local_path, fallback_path)

        logger.warning(f"Used fallback local storage: {fallback_path}")
        return str(fallback_path)

    def download(self, remote_key: str, local_path: Path) -> None:
        """Download and decompress file from S3."""
        full_key = self._get_full_key(remote_key)

        try:
            response = self.client.get_object(
                Bucket=self.config.bucket,
                Key=full_key,
            )
            compressed_data = response["Body"].read()

            # Decompress
            decompressor = zstd.ZstdDecompressor()
            data = decompressor.decompress(compressed_data)

            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(data)

            logger.info(
                f"Downloaded checkpoint from s3://{self.config.bucket}/{full_key}"
            )

        except self.client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Checkpoint not found: {remote_key}")

    def list_objects(self, prefix: str) -> list[dict[str, Any]]:
        """List objects in S3 with given prefix."""
        full_prefix = self._get_full_key(prefix)

        response = self.client.list_objects_v2(
            Bucket=self.config.bucket,
            Prefix=full_prefix,
        )

        objects = []
        for obj in response.get("Contents", []):
            objects.append(
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                }
            )

        return objects

    def delete(self, remote_key: str) -> bool:
        """Delete object from S3."""
        full_key = self._get_full_key(remote_key)

        try:
            self.client.delete_object(
                Bucket=self.config.bucket,
                Key=full_key,
            )
            logger.info(f"Deleted s3://{self.config.bucket}/{full_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from S3: {e}")
            return False

    def exists(self, remote_key: str) -> bool:
        """Check if object exists in S3."""
        full_key = self._get_full_key(remote_key)

        try:
            self.client.head_object(
                Bucket=self.config.bucket,
                Key=full_key,
            )
            return True
        except Exception:
            return False


class GCSBackend(CloudStorageBackend):
    """Google Cloud Storage backend for model checkpoints.

    Uses google-cloud-storage for GCS operations.
    """

    def __init__(self, config: CloudStorageConfig):
        """Initialize GCS backend."""
        self.config = config
        self._client: Any | None = None
        self._bucket: Any | None = None

    @property
    def bucket(self) -> Any:
        """Lazy initialization of GCS bucket."""
        if self._bucket is None:
            try:
                from google.cloud import storage

                if self.config.gcs_credentials_path:
                    self._client = storage.Client.from_service_account_json(
                        self.config.gcs_credentials_path
                    )
                elif self.config.gcs_project:
                    self._client = storage.Client(project=self.config.gcs_project)
                else:
                    self._client = storage.Client()

                self._bucket = self._client.bucket(self.config.bucket)
                logger.info(f"GCS client initialized for bucket: {self.config.bucket}")

            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required for GCS backend. "
                    "Install with: pip install google-cloud-storage"
                )
        return self._bucket

    def _get_blob_name(self, remote_key: str) -> str:
        """Get full blob name including prefix."""
        return f"{self.config.prefix}/{remote_key}".strip("/")

    def upload(
        self,
        local_path: Path,
        remote_key: str,
        metadata: dict[str, str],
    ) -> str:
        """Upload file to GCS with compression."""
        blob_name = self._get_blob_name(remote_key)

        try:
            with open(local_path, "rb") as f:
                data = f.read()

            # Apply zstd compression
            compressor = zstd.ZstdCompressor(level=self.config.compression_level)
            compressed = compressor.compress(data)

            blob = self.bucket.blob(blob_name)
            blob.metadata = metadata
            blob.content_encoding = "zstd"
            blob.upload_from_string(compressed)

            uri = f"gs://{self.config.bucket}/{blob_name}"
            logger.info(f"Uploaded checkpoint to {uri}")
            return uri

        except Exception as e:
            logger.error(f"GCS upload failed: {e}")
            if self.config.fallback_local_path:
                fallback = Path(self.config.fallback_local_path) / remote_key
                fallback.parent.mkdir(parents=True, exist_ok=True)
                import shutil

                shutil.copy2(local_path, fallback)
                return str(fallback)
            raise

    def download(self, remote_key: str, local_path: Path) -> None:
        """Download and decompress file from GCS."""
        blob_name = self._get_blob_name(remote_key)
        blob = self.bucket.blob(blob_name)

        if not blob.exists():
            raise FileNotFoundError(f"Checkpoint not found: {remote_key}")

        compressed_data = blob.download_as_bytes()

        # Decompress
        decompressor = zstd.ZstdDecompressor()
        data = decompressor.decompress(compressed_data)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)

        logger.info(f"Downloaded checkpoint from gs://{self.config.bucket}/{blob_name}")

    def list_objects(self, prefix: str) -> list[dict[str, Any]]:
        """List blobs in GCS with given prefix."""
        full_prefix = self._get_blob_name(prefix)

        blobs = self.bucket.list_blobs(prefix=full_prefix)

        return [
            {
                "key": blob.name,
                "size": blob.size,
                "last_modified": blob.updated.isoformat() if blob.updated else None,
            }
            for blob in blobs
        ]

    def delete(self, remote_key: str) -> bool:
        """Delete blob from GCS."""
        blob_name = self._get_blob_name(remote_key)
        blob = self.bucket.blob(blob_name)

        try:
            blob.delete()
            logger.info(f"Deleted gs://{self.config.bucket}/{blob_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from GCS: {e}")
            return False

    def exists(self, remote_key: str) -> bool:
        """Check if blob exists in GCS."""
        blob_name = self._get_blob_name(remote_key)
        blob = self.bucket.blob(blob_name)
        return blob.exists()


class CloudCheckpointManager:
    """High-level manager for cloud-based model checkpoints.

    Provides a unified interface for saving and loading PyTorch models
    to cloud storage with compression, versioning, and MLflow integration.

    Example:
        >>> config = CloudStorageConfig(bucket="my-models", prefix="nglab")
        >>> manager = CloudCheckpointManager(config, backend="s3")
        >>>
        >>> # Save model
        >>> uri = manager.save_checkpoint(model, "ppo", "1.0.0", metrics={"loss": 0.01})
        >>>
        >>> # Load model
        >>> model = manager.load_checkpoint(model, "ppo", "1.0.0")
    """

    def __init__(
        self,
        config: CloudStorageConfig,
        backend: str = "s3",  # "s3" or "gcs"
    ):
        """Initialize cloud checkpoint manager.

        Args:
            config: Cloud storage configuration.
            backend: Cloud backend to use ("s3" or "gcs").
        """
        self.config = config

        if backend == "s3":
            self._backend: CloudStorageBackend = S3Backend(config)
        elif backend == "gcs":
            self._backend = GCSBackend(config)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 's3' or 'gcs'.")

    def _generate_key(self, model_type: str, version: str) -> str:
        """Generate storage key for model."""
        return f"{model_type}/v{version}/checkpoint.pt.zst"

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        model_type: str,
        version: str,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        metrics: dict[str, float] | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> str:
        """Save model checkpoint to cloud storage.

        Args:
            model: PyTorch model to save.
            model_type: Type of model (e.g., "ppo", "vae").
            version: Version string.
            optimizer: Optimizer state (optional).
            scheduler: Scheduler state (optional).
            metrics: Training metrics to attach.
            extra_data: Additional data to save.

        Returns:
            Cloud URI of saved checkpoint.
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "version": version,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics or {},
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if extra_data is not None:
            checkpoint["extra_data"] = extra_data

        # Save to temp file first
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(checkpoint, f.name)
            temp_path = Path(f.name)

        try:
            remote_key = self._generate_key(model_type, version)
            metadata = {
                "model_type": model_type,
                "version": version,
                "timestamp": checkpoint["timestamp"],
            }

            uri = self._backend.upload(temp_path, remote_key, metadata)
            logger.info(f"Saved checkpoint: {model_type} v{version} -> {uri}")
            return uri

        finally:
            temp_path.unlink()  # Delete temp file

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        model_type: str,
        version: str,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        map_location: str | None = None,
    ) -> dict[str, Any]:
        """Load model checkpoint from cloud storage.

        Args:
            model: Model instance to load state into.
            model_type: Type of model.
            version: Version string.
            optimizer: Optimizer to load state into.
            scheduler: Scheduler to load state into.
            map_location: Device to load tensors to.

        Returns:
            Checkpoint dictionary with metadata.
        """
        remote_key = self._generate_key(model_type, version)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = Path(f.name)

        try:
            self._backend.download(remote_key, temp_path)

            checkpoint = torch.load(temp_path, map_location=map_location)
            model.load_state_dict(checkpoint["model_state_dict"])

            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            logger.info(f"Loaded checkpoint: {model_type} v{version}")
            return checkpoint

        finally:
            temp_path.unlink()

    def list_versions(self, model_type: str) -> list[str]:
        """List all versions for a model type.

        Args:
            model_type: Type of model.

        Returns:
            List of version strings.
        """
        objects = self._backend.list_objects(model_type)

        versions = []
        for obj in objects:
            # Parse version from key: model_type/vX.X.X/checkpoint.pt.zst
            parts = obj["key"].split("/")
            for part in parts:
                if part.startswith("v"):
                    versions.append(part[1:])  # Remove 'v' prefix
                    break

        return sorted(set(versions))

    def delete_checkpoint(self, model_type: str, version: str) -> bool:
        """Delete a checkpoint from cloud storage.

        Args:
            model_type: Type of model.
            version: Version string.

        Returns:
            True if deleted.
        """
        remote_key = self._generate_key(model_type, version)
        return self._backend.delete(remote_key)

    def checkpoint_exists(self, model_type: str, version: str) -> bool:
        """Check if a checkpoint exists.

        Args:
            model_type: Type of model.
            version: Version string.

        Returns:
            True if checkpoint exists.
        """
        remote_key = self._generate_key(model_type, version)
        return self._backend.exists(remote_key)


def create_cloud_manager_from_env(
    backend: str = "s3",
) -> CloudCheckpointManager:
    """Create cloud checkpoint manager from environment variables.

    Environment variables:
        NGLAB_CLOUD_BUCKET: Bucket name (required)
        NGLAB_CLOUD_PREFIX: Object prefix (default: "models")
        AWS_ACCESS_KEY_ID: AWS access key (for S3)
        AWS_SECRET_ACCESS_KEY: AWS secret key (for S3)
        AWS_DEFAULT_REGION: AWS region (for S3)
        GOOGLE_APPLICATION_CREDENTIALS: GCS credentials path (for GCS)
        GCP_PROJECT: GCS project ID (for GCS)

    Args:
        backend: Cloud backend ("s3" or "gcs").

    Returns:
        Configured CloudCheckpointManager.
    """
    bucket = os.environ.get("NGLAB_CLOUD_BUCKET")
    if not bucket:
        raise ValueError("NGLAB_CLOUD_BUCKET environment variable required")

    config = CloudStorageConfig(
        bucket=bucket,
        prefix=os.environ.get("NGLAB_CLOUD_PREFIX", "models"),
        aws_region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        gcs_project=os.environ.get("GCP_PROJECT"),
        gcs_credentials_path=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        fallback_local_path=os.environ.get("NGLAB_FALLBACK_PATH", "model_weights"),
    )

    return CloudCheckpointManager(config, backend=backend)
