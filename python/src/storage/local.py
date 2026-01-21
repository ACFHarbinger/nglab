"""
Local filesystem storage backend for models.
"""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from python.src.storage.base import ModelMetadata, ModelStorage, StorageConfig


class LocalStorage(ModelStorage):
    """Local filesystem storage backend."""

    def __init__(self, config: StorageConfig) -> None:
        """Initialize local filesystem storage backend."""
        super().__init__(config)
        self.base_path = Path(config.local_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, name: str) -> Path:
        """Get directory for a model."""
        return self.base_path / name

    def _version_path(self, name: str, version: str) -> Path:
        """Get path for a specific version."""
        return self._model_dir(name) / f"{version}.pt"

    def _metadata_path(self, name: str, version: str) -> Path:
        """Get metadata path for a version."""
        return self._model_dir(name) / f"{version}.json"

    def _latest_path(self, name: str) -> Path:
        """Get path to latest version symlink/file."""
        return self._model_dir(name) / "latest"

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
        """Save model to local filesystem."""
        if version is None:
            version = self._generate_version()

        model_dir = self._model_dir(name)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Compress and save model data
        compressed_data = self._compress(model_data)
        model_path = self._version_path(name, version)
        with open(model_path, "wb") as f:
            f.write(compressed_data)

        # Save metadata
        meta = ModelMetadata(
            name=name,
            version=version,
            checksum=self.compute_checksum(model_data),
            size_bytes=len(model_data),
            created_at=datetime.now(UTC).isoformat(),
            **(metadata or {}),
        )
        with open(self._metadata_path(name, version), "w") as f:
            json.dump(meta.__dict__, f, indent=2)

        # Update latest pointer
        latest_path = self._latest_path(name)
        with open(latest_path, "w") as f:
            f.write(version)

        # Clean up old versions if needed
        if self.config.versioning:
            self._cleanup_old_versions(name)

        return str(model_path)

    def load(self, name: str, version: str | None = None) -> bytes:
        """Load model from local filesystem."""
        if version is None:
            version = self._get_latest_version(name)
            if version is None:
                raise FileNotFoundError(f"Model '{name}' not found")

        model_path = self._version_path(name, version)
        if not model_path.exists():
            raise FileNotFoundError(f"Model '{name}' version '{version}' not found")

        with open(model_path, "rb") as f:
            compressed_data = f.read()

        return self._decompress(compressed_data)

    def exists(self, name: str, version: str | None = None) -> bool:
        """Check if model exists."""
        if version is None:
            return self._model_dir(name).exists()
        return self._version_path(name, version).exists()

    def delete(self, name: str, version: str | None = None) -> bool:
        """Delete model from storage."""
        if version is None:
            # Delete entire model directory
            model_dir = self._model_dir(name)
            if model_dir.exists():
                shutil.rmtree(model_dir)
                return True
            return False
        else:
            # Delete specific version
            model_path = self._version_path(name, version)
            meta_path = self._metadata_path(name, version)
            deleted = False
            if model_path.exists():
                model_path.unlink()
                deleted = True
            if meta_path.exists():
                meta_path.unlink()
            return deleted

    def list_models(self) -> list[str]:
        """List all model names."""
        if not self.base_path.exists():
            return []
        return [
            d.name
            for d in self.base_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    def list_versions(self, name: str) -> list[str]:
        """List all versions of a model."""
        model_dir = self._model_dir(name)
        if not model_dir.exists():
            return []
        return sorted(
            [f.stem for f in model_dir.glob("*.pt") if f.stem != "latest"],
            reverse=True,
        )

    def get_metadata(self, name: str, version: str | None = None) -> ModelMetadata:
        """Get metadata for a model."""
        if version is None:
            version = self._get_latest_version(name)
            if version is None:
                raise FileNotFoundError(f"Model '{name}' not found")

        meta_path = self._metadata_path(name, version)
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata for '{name}' version '{version}' not found"
            )

        with open(meta_path) as f:
            data = json.load(f)

        return ModelMetadata(**data)

    def _get_latest_version(self, name: str) -> str | None:
        """Get the latest version string for a model."""
        latest_path = self._latest_path(name)
        if latest_path.exists():
            with open(latest_path) as f:
                return f.read().strip()

        # Fallback: find most recent version file
        versions = self.list_versions(name)
        return versions[0] if versions else None

    def _cleanup_old_versions(self, name: str) -> None:
        """Remove old versions beyond max_versions."""
        versions = self.list_versions(name)
        if len(versions) > self.config.max_versions:
            for old_version in versions[self.config.max_versions :]:
                self.delete(name, old_version)
