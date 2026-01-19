"""
Model Storage Module for NGLab.

Provides unified interface for storing and retrieving model checkpoints
across different backends: local filesystem, AWS S3, and Google Cloud Storage.
"""

from python.src.storage.base import ModelStorage, StorageConfig
from python.src.storage.local import LocalStorage
from python.src.storage.s3 import S3Storage
from python.src.storage.gcs import GCSStorage
from python.src.storage.factory import create_storage

__all__ = [
    "ModelStorage",
    "StorageConfig",
    "LocalStorage",
    "S3Storage",
    "GCSStorage",
    "create_storage",
]
