"""Concept drift detection utilities."""

from abc import ABC, abstractmethod

import numpy as np


class DriftDetector(ABC):
    """Abstract base class for concept drift detection."""

    def __init__(self) -> None:
        """Initialize DriftDetector."""
        self.in_drift = False
        self.in_warning = False

    @abstractmethod
    def update(self, value: float) -> bool:
        """
        Update the detector with a new value.
        Returns True if drift is detected.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the detector state."""
        pass


class PageHinkley(DriftDetector):
    """
    Page-Hinkley Test (PHT) for concept drift detection.

    Detects changes in the average of a signal.
    """

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
    ) -> None:
        """
        Args:
            min_instances: Minimum number of instances before detecting drift.
            delta: Minimum magnitude of change to track.
            threshold: Threshold for drift detection.
            alpha: Decay factor for the mean (forgetting factor).
        """
        super().__init__()
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha

        self.sample_count = 0
        self.x_mean = 0.0
        self.sum_upper = 0.0
        self.sum_lower = 0.0

    def reset(self) -> None:
        """Reset Page-Hinkley detector state."""
        self.sample_count = 0
        self.x_mean = 0.0
        self.sum_upper = 0.0
        self.sum_lower = 0.0
        self.in_drift = False

    def update(self, value: float) -> bool:
        """Update Page-Hinkley detector with a new value."""
        self.sample_count += 1

        if self.sample_count == 1:
            self.x_mean = value
            return False

        # Update running mean
        # self.x_mean = self.x_mean + (value - self.x_mean) / self.sample_count # Standard
        # Use fading factor for mean to adapt to slow trends better before jumping
        self.x_mean = self.alpha * self.x_mean + (1 - self.alpha) * value

        # Upper Drift (Increase)
        self.sum_upper = max(0, self.sum_upper + (value - self.x_mean - self.delta))

        # Lower Drift (Decrease)
        self.sum_lower = max(0, self.sum_lower + (self.x_mean - value - self.delta))

        if self.sample_count < self.min_instances:
            return False

        if self.sum_upper > self.threshold or self.sum_lower > self.threshold:
            self.in_drift = True
            return True

        self.in_drift = False
        return False


class MovingAverageDrift(DriftDetector):
    """
    Simple drift detector comparing Short-Term vs Long-Term moving averages.
    """

    def __init__(
        self, short_window: int = 20, long_window: int = 100, threshold: float = 3.0
    ) -> None:
        """Initialize MovingAverageDrift detector."""
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.threshold = threshold  # Z-score like threshold or absolute diff percent

        self.buffer: list[float] = []

    def reset(self) -> None:
        """Reset moving average detector state."""
        self.buffer = []
        self.in_drift = False

    def update(self, value: float) -> bool:
        """Update moving average detector with a new value."""
        self.buffer.append(value)
        if len(self.buffer) > self.long_window:
            self.buffer.pop(0)

        if len(self.buffer) < self.long_window:
            return False

        long_ma = float(np.mean(self.buffer))
        short_ma = float(np.mean(self.buffer[-self.short_window :]))
        std_dev = float(np.std(self.buffer)) + 1e-6

        z_score = abs(short_ma - long_ma) / std_dev

        if z_score > self.threshold:
            self.in_drift = True
            return True

        self.in_drift = False
        return False
