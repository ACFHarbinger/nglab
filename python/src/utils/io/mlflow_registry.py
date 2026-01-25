"""
MLflow model registry integration for NGLab.

Provides ModelRegistry class for:
- Logging models with metrics and hyperparameters
- Loading versioned models
- Managing model lifecycle stages (Staging, Production)
"""

import logging
from typing import Any

import mlflow
from mlflow import pytorch, tracking

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    MLflow-based model versioning and registry.

    Handles logging of hyperparameters, metrics, and PyTorch models to a
    centralized tracking server or local storage.
    """

    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db"):
        """Initialize MLflow tracking URI."""
        try:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracing initialized with URI: {tracking_uri}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")

    def log_model(  # noqa: PLR0913
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: str,
        metrics: dict[str, float],
        hyperparameters: dict[str, Any],
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        Log a model with its evaluation metrics and hyperparameters.

        Args:
            model: The PyTorch model instance to log.
            artifact_path: Local path or relative name for the model artifact.
            registered_model_name: Name of the model in the MLflow Registry.
            metrics: Dictionary of numerical metrics (e.g., loss, accuracy).
            hyperparameters: Dictionary of configuration parameters.
            tags: Optional metadata tags.
        """
        try:
            with mlflow.start_run():
                # Log hyperparameters
                mlflow.log_params(hyperparameters)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Log tags
                if tags:
                    mlflow.set_tags(tags)

                # Log model using PyTorch flavor
                pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                )
                logger.info(f"Model '{registered_model_name}' logged successfully.")
        except Exception as e:
            logger.error(f"Failed to log model to MLflow: {e}")

    def load_model(self, model_name: str, version: str = "latest") -> Any:
        """
        Load a registered model from the registry.

        Args:
            model_name: Name of the model in the registry.
            version: Specific version number or 'latest'.
        """
        model_uri = f"models:/{model_name}/{version}"
        try:
            model = pytorch.load_model(model_uri)
            logger.info(f"Model '{model_name}' version '{version}' loaded.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            raise

    def transition_model_stage(self, model_name: str, version: int, stage: str) -> None:
        """
        Transition a model version to a different stage (e.g., 'Staging', 'Production').
        """
        try:
            client = tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=str(version),
                stage=stage,
            )
            logger.info(f"Model '{model_name}' v{version} transitioned to '{stage}'.")
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
