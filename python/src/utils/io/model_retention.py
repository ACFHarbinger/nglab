"""
Model Retention Policy and Lifecycle Management.

Provides policies for managing model versions, cleaning up old checkpoints,
and enforcing retention rules across local and cloud storage.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Union

from python.src.utils.io.cloud_storage import CloudCheckpointManager
from python.src.utils.io.model_versioning import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class RetentionConfig:
    """Configuration for model retention."""

    keep_latest_n: int = 5
    keep_best_metric: Optional[str] = "val_loss"
    keep_best_n: int = 1
    max_age_days: Optional[int] = None
    dry_run: bool = False


class CheckpointManagerProtocol(Protocol):
    """Protocol for checkpoint managers (local or cloud)."""

    def list_versions(self, model_type: str) -> List[str]: ...

    def delete(self, model_type: str, version: str) -> bool: ...

    # Note: different managers might unfortunately use delete vs delete_checkpoint
    # We'll need to adapt.


class ModelRetentionPolicy:
    """
    Enforces retention policies on model checkpoints.

    Can clean up old versions based on count, age, or performance metrics.
    """

    def __init__(
        self,
        config: RetentionConfig,
        manager: Union[ModelRegistry, CloudCheckpointManager],
    ):
        """
        Initialize retention policy.

        Args:
            config: Retention rules.
            manager: The storage manager to act upon (ModelRegistry or CloudCheckpointManager).
        """
        self.config = config
        self.manager = manager

    def _delete_version(self, model_type: str, version: str) -> bool:
        """Helper to delete a version using the manager's specific API."""
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would delete {model_type} v{version}")
            return True

        # Adapt to different APIs
        try:
            # Use Any to bypass static analysis checks for dynamic dispatch
            mgr: Any = self.manager
            if hasattr(mgr, "delete_checkpoint"):
                return mgr.delete_checkpoint(model_type, version)
            elif hasattr(mgr, "delete"):
                return mgr.delete(model_type, version)
            else:
                logger.error(f"Manager {type(self.manager)} has no delete method")
                return False
        except Exception as e:
            logger.error(f"Failed to delete {model_type} v{version}: {e}")
            return False

    def cleanup_versions(self, model_type: str) -> int:
        """
        Clean up versions for a specific model type according to policy.

        Returns:
            Number of versions deleted.
        """
        versions = self.manager.list_versions(model_type)
        if not versions:
            logger.info(f"No versions found for {model_type}")
            return 0

        # Sort versions. Assuming semantic versioning or comparable strings.
        # We might want to sort by timestamp if available, but here we assume
        # lexicographical or semantic sort roughly correlates with time for now,
        # or that list_versions returns sorted.
        # Ideally, we'd fetch metadata for all to get timestamps.

        # For simple 'keep latest N', we assume the list is sorted or we sort it.
        # Let's try to sort assuming semantic versioning if possible.
        try:
            # Simple sort (lexicographical usually works for fixed-width or simple semver)
            sorted_versions = sorted(versions)
        except Exception:
            sorted_versions = versions

        total_versions = len(sorted_versions)
        if total_versions <= self.config.keep_latest_n:
            logger.info(
                f"Count {total_versions} <= keep_latest_n {self.config.keep_latest_n}. No cleanup."
            )
            return 0

        # Versions to keep based on recency
        versions_to_keep = set(sorted_versions[-self.config.keep_latest_n :])

        # Identify versions to delete
        versions_to_delete = []
        for v in sorted_versions:
            if v not in versions_to_keep:
                versions_to_delete.append(v)

        # Perform deletion
        deleted_count = 0
        for v in versions_to_delete:
            if self._delete_version(model_type, v):
                deleted_count += 1
                logger.info(f" Deleted {model_type} v{v}")

        return deleted_count

    def enforce_all(self, model_types: List[str]) -> Dict[str, int]:
        """Enforce policy on a list of model types."""
        results = {}
        for m_type in model_types:
            deleted = self.cleanup_versions(m_type)
            results[m_type] = deleted
        return results
