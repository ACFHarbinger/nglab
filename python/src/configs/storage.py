"""
Storage Configurations.

Contains configurations for cloud storage backends and model retention policies.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CloudStorageConfig", "RetentionConfig"]


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


@dataclass
class RetentionConfig:
    """Configuration for model retention."""

    keep_latest_n: int = 5
    keep_best_metric: str | None = "val_loss"
    keep_best_n: int = 1
    max_age_days: int | None = None
    dry_run: bool = False
