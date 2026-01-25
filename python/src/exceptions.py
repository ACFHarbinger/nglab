from __future__ import annotations

class NGLabError(Exception):
    """Base exception for NGLab."""
    pass

class ConfigurationError(NGLabError):
    """Raised when configuration is invalid."""
    pass

class ModelNotFoundError(NGLabError):
    """Raised when a model cannot be found."""
    pass

class EnvironmentError(NGLabError):
    """Raised when environment encounters an error."""
    pass

class TrainingError(NGLabError):
    """Raised when a training process fails."""
    pass

class DataError(NGLabError):
    """Raised when data loading or processing fails."""
    pass
