"""
Model versioning and artifact management for NGLab.

This module provides utilities for saving, loading, and managing model versions
with comprehensive metadata for reproducibility and experiment tracking.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn


@dataclass
class ModelMetadata:
    """Metadata for a saved model checkpoint.

    Attributes:
        version: Semantic version string (e.g., "1.0.0")
        model_type: Model architecture type (e.g., "VAE", "PPO", "SAC")
        training_config: Dictionary of training parameters
        metrics: Dictionary of evaluation metrics
        framework_version: Framework version string (e.g., "pytorch-2.3.0")
        training_date: ISO format timestamp
        dataset_hash: Hash of training dataset for reproducibility
        git_commit: Git commit hash when model was trained (optional)
        description: Human-readable description (optional)
        dependencies: Dictionary of dependency versions (optional)
        tags: List of tags for organization (optional)
    """

    version: str
    model_type: str
    training_config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    framework_version: str = field(
        default_factory=lambda: f"pytorch-{torch.__version__}"
    )
    training_date: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_hash: str = "unknown"
    git_commit: str | None = None
    description: str | None = None
    dependencies: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ModelMetadata":
        """Create metadata from JSON string."""
        return cls.from_dict(json.loads(json_str))


def compute_dataset_hash(data_path: str | Path) -> str:
    """Compute SHA256 hash of dataset for reproducibility.

    Args:
        data_path: Path to dataset file or directory

    Returns:
        Hexadecimal hash string
    """
    data_path = Path(data_path)
    hasher = hashlib.sha256()

    if data_path.is_file():
        with open(data_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
    elif data_path.is_dir():
        # Hash all files in directory
        for file_path in sorted(data_path.rglob("*")):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
    else:
        raise ValueError(f"Invalid data path: {data_path}")

    return hasher.hexdigest()


def get_git_commit() -> str | None:
    """Get current git commit hash.

    Returns:
        Git commit hash or None if not in a git repository
    """
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def save_model_with_metadata(
    model: nn.Module,
    save_path: str | Path,
    metadata: ModelMetadata,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> None:
    """Save model checkpoint with comprehensive metadata.

    Args:
        model: PyTorch model to save
        save_path: Path to save checkpoint
        metadata: Model metadata
        optimizer: Optimizer state (optional)
        scheduler: Learning rate scheduler state (optional)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata.to_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)

    # Also save metadata as separate JSON for easy inspection
    metadata_path = save_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        f.write(metadata.to_json())


def load_model_with_metadata(
    model: nn.Module,
    load_path: str | Path,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> tuple[nn.Module, ModelMetadata]:
    """Load model checkpoint with metadata.

    Args:
        model: Model instance to load state into
        load_path: Path to checkpoint
        map_location: Device to load tensors to
        strict: Whether to strictly enforce state_dict keys match

    Returns:
        Tuple of (loaded_model, metadata)
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    checkpoint = cast(dict[str, Any], torch.load(load_path, map_location=map_location))

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    metadata = ModelMetadata.from_dict(checkpoint["metadata"])

    return model, metadata


def check_version_compatibility(current_version: str, checkpoint_version: str) -> bool:
    """Check if checkpoint version is compatible with current code.

    Uses semantic versioning: major.minor.patch
    - Major version mismatch: incompatible
    - Minor/patch mismatch: compatible with warning

    Args:
        current_version: Current model version
        checkpoint_version: Checkpoint version

    Returns:
        True if compatible, False otherwise
    """

    def parse_version(version: str) -> tuple[int, int, int]:
        """Parse version string into tuple of integers."""
        parts = version.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version}")
        v_tuple = tuple(map(int, parts))
        return cast(tuple[int, int, int], v_tuple)

    current = parse_version(current_version)
    checkpoint = parse_version(checkpoint_version)

    # Major version must match
    return current[0] == checkpoint[0]


class ModelRegistry:
    """Registry for managing multiple model versions.

    Provides a centralized interface for saving, loading, and listing models.
    """

    def __init__(self, base_path: str | Path) -> None:
        """Initialize model registry.

        Args:
            base_path: Base directory for storing models
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_model_path(self, model_type: str, version: str) -> Path:
        """Get path for a specific model version."""
        return self.base_path / model_type / f"v{version}.pt"

    def save(
        self,
        model: nn.Module,
        model_type: str,
        version: str,
        metadata: ModelMetadata,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> Path:
        """Save model to registry.

        Args:
            model: Model to save
            model_type: Type of model (e.g., "vae", "ppo")
            version: Version string
            metadata: Model metadata
            optimizer: Optimizer state (optional)

        Returns:
            Path where model was saved
        """
        save_path = self._get_model_path(model_type, version)
        save_model_with_metadata(model, save_path, metadata, optimizer)
        return save_path

    def load(
        self,
        model: nn.Module,
        model_type: str,
        version: str,
        map_location: str | torch.device | None = None,
        strict: bool = True,
    ) -> tuple[nn.Module, ModelMetadata]:
        """Load model from registry.

        Args:
            model: Model instance to load state into
            model_type: Type of model
            version: Version string
            map_location: Device to load to

        Returns:
            Tuple of (loaded_model, metadata)
        """
        load_path = self._get_model_path(model_type, version)
        return load_model_with_metadata(model, load_path, map_location, strict=strict)

    def list_versions(self, model_type: str) -> list[str]:
        """List all available versions for a model type.

        Args:
            model_type: Type of model

        Returns:
            List of version strings
        """
        model_dir = self.base_path / model_type
        if not model_dir.exists():
            return []

        versions = []
        for path in model_dir.glob("v*.pt"):
            version = path.stem[1:]  # Remove 'v' prefix
            versions.append(version)

        return sorted(versions, reverse=True)  # Most recent first

    def get_latest(self, model_type: str) -> str | None:
        """Get the latest version for a model type.

        Args:
            model_type: Type of model

        Returns:
            Latest version string or None if no versions exist
        """
        versions = self.list_versions(model_type)
        return versions[0] if versions else None

    def get_metadata(self, model_type: str, version: str) -> ModelMetadata:
        """Get metadata for a specific model version.

        Args:
            model_type: Type of model
            version: Version string

        Returns:
            ModelMetadata instance
        """
        _, metadata = self.load(nn.Module(), model_type, version, strict=False)
        return metadata

    def delete(self, model_type: str, version: str) -> bool:
        """Delete a model version from registry.

        Args:
            model_type: Type of model
            version: Version string

        Returns:
            True if deleted, False if not found
        """
        model_path = self._get_model_path(model_type, version)
        metadata_path = model_path.with_suffix(".json")

        deleted = False
        if model_path.exists():
            model_path.unlink()
            deleted = True
        if metadata_path.exists():
            metadata_path.unlink()

        return deleted


def create_metadata_from_config(  # noqa: PLR0913
    model_type: str,
    config: dict[str, Any],
    metrics: dict[str, float],
    dataset_path: str | Path | None = None,
    version: str = "1.0.0",
    description: str | None = None,
    tags: list[str] | None = None,
) -> ModelMetadata:
    """Create model metadata from training configuration.

    Convenience function for creating metadata during training.

    Args:
        model_type: Model architecture type
        config: Configuration dictionary with hyperparameters
        metrics: Evaluation metrics
        dataset_path: Path to training dataset (optional)
        version: Version string (default: "1.0.0")
        description: Model description (optional)
        tags: List of tags (optional)

    Returns:
        ModelMetadata instance
    """
    dataset_hash = compute_dataset_hash(dataset_path) if dataset_path else "unknown"

    return ModelMetadata(
        version=version,
        model_type=model_type,
        framework_version=f"pytorch-{torch.__version__}",
        training_config=config,
        metrics=metrics,
        training_date=datetime.now().isoformat(),
        dataset_hash=dataset_hash,
        git_commit=get_git_commit(),
        description=description,
        tags=tags or [],
    )
