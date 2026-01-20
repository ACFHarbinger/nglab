"""
Market Regime Detection for Meta-Learning.

Detects different market conditions (volatile, trending, ranging) to trigger
model adaptation.
"""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.cluster import KMeans


class RegimeDetector:
    """
    Detects market regimes based on statistical features.

    Uses clustering on volatility, trend, and volume features.
    """

    def __init__(self, n_regimes: int = 3, window_size: int = 50) -> None:
        """
        Initialize regime detector.

        Args:
            n_regimes: Number of market regimes to detect.
            window_size: Rolling window for feature calculation.
        """
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.kmeans: KMeans | None = None
        self.regime_labels = ["Volatile", "Trending", "Ranging"][:n_regimes]

    def extract_features(self, prices: NDArray[Any]) -> NDArray[Any]:
        """
        Extract market features for regime classification.

        Args:
            prices: Price series [num_samples].

        Returns:
            Features array [num_windows, num_features].
        """
        features = []

        for i in range(len(prices) - self.window_size):
            window = prices[i : i + self.window_size]

            # Volatility (std of returns)
            returns = np.diff(window) / window[:-1]
            volatility = np.std(returns)

            # Trend (linear regression slope)
            x = np.arange(len(window))
            trend = np.polyfit(x, window, 1)[0]

            # Range (max - min) / mean
            range_pct = (np.max(window) - np.min(window)) / np.mean(window)

            features.append([volatility, trend, range_pct])

        return np.array(features)

    def fit(self, prices: NDArray[Any]) -> None:
        """
        Fit regime detector on historical price data.

        Args:
            prices: Historical prices.
        """
        features = self.extract_features(prices)
        self.kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        self.kmeans.fit(features)

    def predict(self, prices: NDArray[Any]) -> int:
        """
        Predict current market regime.

        Args:
            prices: Recent price history (at least window_size).

        Returns:
            Regime ID (0 to n_regimes-1).
        """
        if self.kmeans is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        if len(prices) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} prices")

        # Use most recent window
        window = prices[-self.window_size :]
        features = self.extract_features(
            np.concatenate([prices[-2 * self.window_size :], window])
        )

        regime = self.kmeans.predict(features[-1:])
        return int(regime[0])

    def get_regime_name(self, regime_id: int) -> str:
        """Get human-readable name for regime."""
        return self.regime_labels[regime_id]

    def partition_by_regime(
        self, prices: NDArray[Any], data: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """
        Partition data by detected regimes.

        Args:
            prices: Price series for regime detection.
            data: Corresponding data to partition.

        Returns:
            Dict mapping regime_id to data subset.
        """
        features = self.extract_features(prices)
        if self.kmeans is None:
            self.fit(prices)

        if self.kmeans is None:
            raise ValueError("KMeans failed to initialize")

        regimes = self.kmeans.predict(features)

        partitions: dict[int, torch.Tensor] = {}
        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            partitions[regime_id] = data[mask]

        return partitions
