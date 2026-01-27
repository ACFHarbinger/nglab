from __future__ import annotations

import numpy as np
from typing import Any


class OnlineNormalizer:
    """
    Online normalization using Welford's Online Algorithm for mean and variance.
    Useful for real-time streaming data where full history is unavailable.
    """

    def __init__(self, feature_dim: int, epsilon: float = 1e-8):
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        
        self.n = 0
        self.mean = np.zeros(feature_dim)
        self.m2 = np.zeros(feature_dim)  # Sum of squares of differences from the mean

    def update(self, x: np.ndarray) -> None:
        """Update running mean and variance with a new observation."""
        x = np.asarray(x)
        if x.shape[-1] != self.feature_dim:
            raise ValueError(f"Input dimension {x.shape[-1]} does not match feature_dim {self.feature_dim}")
            
        # Welford's Algorithm
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Scale the input using current mean and standard deviation."""
        if self.n < 2:
            return x - self.mean
            
        variance = self.m2 / self.n
        std = np.sqrt(variance)
        return (x - self.mean) / (std + self.epsilon)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Update and transform in one step."""
        self.update(x)
        return self.transform(x)

    @property
    def std(self) -> np.ndarray:
        """Current running standard deviation."""
        if self.n < 2:
            return np.zeros(self.feature_dim)
        return np.sqrt(self.m2 / self.n)
