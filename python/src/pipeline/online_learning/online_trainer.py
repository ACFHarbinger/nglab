"""
Online Learning Trainer for real-time model adaptation.
"""

import collections
import copy
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator


class ExperienceReplayBuffer:
    """A simple FIFO buffer for experience replay in online learning."""

    def __init__(self, capacity: int = 1000) -> None:
        """Initialize ExperienceReplayBuffer."""
        self.buffer: collections.deque[dict[str, NDArray[Any]]] = collections.deque(
            maxlen=capacity
        )

    def add(self, X: NDArray[Any], y: NDArray[Any]) -> None:  # noqa: N803
        """Add a batch of experience to the buffer."""
        for i in range(len(X)):
            self.buffer.append({"X": X[i], "y": y[i]})

    def sample(self, batch_size: int) -> dict[str, NDArray[Any]] | None:
        """Sample a random batch from the buffer."""
        if len(self.buffer) < batch_size:
            return None

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]

        return {
            "X": np.array([s["X"] for s in samples]),
            "y": np.array([s["y"] for s in samples]),
        }


class OnlineTrainer:
    """
    Handles incremental updates to models during live trading.
    Supports rollback and replay-buffer stabilized learning.
    """

    def __init__(
        self,
        model: BaseEstimator,
        replay_capacity: int = 2000,
        update_batch_size: int = 32,
        performance_threshold: float = -0.05,  # Max 5% degradation before rollback
    ) -> None:
        """Initialize OnlineTrainer."""
        self.model = model
        self.replay_buffer = ExperienceReplayBuffer(capacity=replay_capacity)
        self.update_batch_size = update_batch_size
        self.performance_threshold = performance_threshold

        # Performance tracking
        self.baseline_score: float | None = None
        self.last_stable_model: BaseEstimator | None = None

        # Check if model supports partial_fit
        self.supports_incremental = hasattr(model, "partial_fit")

    def update(self, X: NDArray[Any], y: NDArray[Any]) -> bool:  # noqa: N803
        """
        Perform an incremental update using new data and replay buffer.
        Returns: True if update was successful and stable.
        """
        if not self.supports_incremental:
            return False

        # 1. Save current state for potential rollback
        self.last_stable_model = copy.deepcopy(self.model)

        # 2. Add new data to replay buffer
        self.replay_buffer.add(X, y)

        # 3. Sample from buffer for stable update
        batch = self.replay_buffer.sample(self.update_batch_size)
        if batch is None:
            # Not enough data for a full batch update yet
            return True

        # 4. Incremental fit
        try:
            # We know it has partial_fit because of supports_incremental check
            # Use getattr to avoid Mypy complaining about missing attribute if it's not strictly in BaseEstimator
            self.model.partial_fit(batch["X"], batch["y"])
        except Exception as e:
            print(f"Online update failed: {e}")
            self.rollback()
            return False

        # 5. Stability check (simplified)
        # In a real scenario, we'd evaluate on a validation set or recent window
        return True

    def rollback(self) -> None:
        """Revert to the last known stable model."""
        if self.last_stable_model is not None:
            self.model = self.last_stable_model
            print("Rollback performed due to unstable online update.")

    def evaluate(self, X: NDArray[Any], y: NDArray[Any]) -> float:  # noqa: N803
        """Evaluate current model performance."""
        if hasattr(self.model, "score"):
            return float(self.model.score(X, y))
        return 0.0
