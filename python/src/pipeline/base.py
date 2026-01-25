
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePipeline(ABC):
    """Base class for all pipelines."""

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Run the pipeline."""


class BaseTrainer(BasePipeline):
    """Base trainer interface."""

    @abstractmethod
    def train(self) -> Any:
        """Train the model."""

    def run(self, **kwargs: Any) -> Any:
        return self.train()


class BaseEvaluator(BasePipeline):
    """Base evaluator interface."""

    @abstractmethod
    def evaluate(self) -> Any:
        """Evaluate the model."""

    def run(self, **kwargs: Any) -> Any:
        return self.evaluate()
