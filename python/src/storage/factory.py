"""
Factory function for creating storage backends.
"""

from python.src.storage.base import ModelStorage, StorageConfig
from python.src.storage.gcs import GCSStorage
from python.src.storage.local import LocalStorage
from python.src.storage.s3 import S3Storage


def create_storage(config: StorageConfig | None = None) -> ModelStorage:
    """
    Create a storage backend based on configuration.

    Args:
        config: Storage configuration. If None, creates from environment.

    Returns:
        Configured ModelStorage instance.

    Raises:
        ValueError: If storage_type is not recognized.
    """
    if config is None:
        config = StorageConfig.from_env()

    storage_type = config.storage_type.lower()

    if storage_type == "local":
        return LocalStorage(config)
    elif storage_type == "s3":
        if not config.s3_bucket:
            raise ValueError("S3 bucket name is required for S3 storage")
        return S3Storage(config)
    elif storage_type == "gcs":
        if not config.gcs_bucket:
            raise ValueError("GCS bucket name is required for GCS storage")
        return GCSStorage(config)
    else:
        raise ValueError(
            f"Unknown storage type: {storage_type}. Supported types: local, s3, gcs"
        )
