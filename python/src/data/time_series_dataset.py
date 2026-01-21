"""
Time Series Dataset for sliding window training.

Loads CSV data and creates (X, Y) pairs for supervised learning.
"""

from typing import Literal

import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Dataset for time series forecasting with sliding windows.

    Args:
        csv_path: Path to CSV file.
        target_column: Column name to predict.
        seq_len: Length of input sequence (lookback).
        pred_len: Length of prediction horizon.
        train: Whether this is training split.
        train_ratio: Ratio of data for training (rest is validation).
        normalize: Normalization method ('minmax', 'zscore', or None).
    """

    def __init__(  # noqa: PLR0913
        self,
        csv_path: str,
        target_column: str,
        seq_len: int = 30,
        pred_len: int = 1,
        train: bool = True,
        train_ratio: float = 0.8,
        normalize: Literal["minmax", "zscore"] | None = "minmax",
        stats: dict[str, float] | None = None,
    ):
        """Initialize TimeSeriesDataset."""
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.normalize = normalize

        # Load and parse CSV
        df = pd.read_csv(csv_path)

        # Find target column
        if target_column not in df.columns:
            raise ValueError(
                f"Column '{target_column}' not found. Available: {list(df.columns)}"
            )

        # Extract target values
        values = df[target_column].astype(float).to_numpy()

        # Split train/val
        split_idx = int(len(values) * train_ratio)
        if train:
            values = values[:split_idx]
        else:
            values = values[split_idx:]

        # Store raw stats for denormalization
        if stats:
            self.raw_min = stats.get("min", float(values.min()))
            self.raw_max = stats.get("max", float(values.max()))
            self.raw_mean = stats.get("mean", float(values.mean()))
            self.raw_std = stats.get("std", float(values.std()))
        else:
            self.raw_min = float(values.min())
            self.raw_max = float(values.max())
            self.raw_mean = float(values.mean())
            self.raw_std = float(values.std())

        # Normalize
        if normalize == "minmax":
            if self.raw_max - self.raw_min > 1e-8:
                values = (values - self.raw_min) / (self.raw_max - self.raw_min)
            else:
                values = values - self.raw_min
        elif normalize == "zscore":
            if self.raw_std > 1e-8:
                values = (values - self.raw_mean) / self.raw_std
            else:
                values = values - self.raw_mean

        self.data = torch.tensor(values, dtype=torch.float32)

    def __len__(self) -> int:
        """Number of sliding window samples."""
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a sample.

        Returns:
            dict with:
                - observation: (seq_len, 1) tensor
                - target: (pred_len,) tensor
        """
        x = self.data[idx : idx + self.seq_len].unsqueeze(-1)  # (seq_len, 1)
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]

        return {"observation": x, "target": y}

    def get_columns(self) -> list[str]:
        """Return available column names from CSV (class method)."""
        return []  # Not implemented as instance method

    @staticmethod
    def list_columns(csv_path: str) -> list[str]:
        """
        List available columns in a CSV file.

        Args:
            csv_path: Path to CSV file.

        Returns:
            List of column names (excluding date/time columns).
        """
        df = pd.read_csv(csv_path, nrows=0)
        # Filter out common date/time columns
        exclude_lower = {"date", "time", "timestamp", "date (utc)", "timestamp (utc)"}
        return [c for c in df.columns if c.lower() not in exclude_lower]
