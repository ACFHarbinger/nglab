
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["BasePipeline", "BaseTrainer", "BaseEvaluator", "BaseCallback"]


class BasePipeline(ABC):
    """Base class for all pipelines."""

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Run the pipeline."""
        pass


class BaseTrainer(BasePipeline):
    """Base trainer interface."""

    @abstractmethod
    def train(self, **kwargs: Any) -> Any:
        """Train the model."""
        pass

    def run(self, **kwargs: Any) -> Any:
        return self.train(**kwargs)


class BaseEvaluator(BasePipeline):
    """Base evaluator interface."""

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> Any:
        """Evaluate the model."""
        pass

    def run(self, **kwargs: Any) -> Any:
        return self.evaluate(**kwargs)


class BaseCallback(ABC):
    """Base class for all training callbacks."""

    def on_train_begin(self, **kwargs: Any) -> None:
        """Called when training begins."""
        pass

    def on_train_end(self, **kwargs: Any) -> None:
        """Called when training ends."""
        pass

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        """Called when an epoch begins."""
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
        """Called when an epoch ends."""
        pass

    def on_batch_begin(self, batch_idx: int, **kwargs: Any) -> None:
        """Called when a batch begins."""
        pass

    def on_batch_end(self, batch_idx: int, metrics: dict[str, float], **kwargs: Any) -> None:
        """Called when a batch ends."""
        pass
