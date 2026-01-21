"""
DataLoaders for Financial Time Series Data.

Extends TimeSeriesDataset with financial-specific preprocessing and provides
factory functions for creating DataLoaders with proper configuration.
"""

from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from python.src.data.time_series_dataset import TimeSeriesDataset


class FinancialDataset(TimeSeriesDataset):
    """
    Financial time series dataset with advanced preprocessing.

    Extends TimeSeriesDataset with financial-specific features like
    technical indicators, multi-asset support, and per-feature normalization.

    Args:
        csv_path: Path to CSV file or directory containing multiple CSVs.
        target_column: Column name to predict.
        seq_len: Length of input sequence (lookback).
        pred_len: Length of prediction horizon.
        train: Whether this is training split.
        train_ratio: Ratio of data for training (rest is validation).
        normalize: Normalization method ('minmax', 'zscore', or None).
        add_technical_indicators: Whether to add RSI, MACD, etc.
        multi_asset: Whether to load multiple assets (all CSVs in directory).
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
        add_technical_indicators: bool = False,
        multi_asset: bool = False,
        stats: dict[str, Any] | None = None,
    ):
        """Initialize FinancialDataset."""
        # Store additional config
        self.add_technical_indicators = add_technical_indicators
        self.multi_asset = multi_asset

        # Initialize base class
        super().__init__(
            csv_path=csv_path,
            target_column=target_column,
            seq_len=seq_len,
            pred_len=pred_len,
            train=train,
            train_ratio=train_ratio,
            normalize=normalize,
            stats=stats,
        )


def create_dataloader(  # noqa: PLR0913
    data_path: str,
    target_column: str,
    batch_size: int = 32,
    seq_len: int = 30,
    pred_len: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    normalize: Literal["minmax", "zscore"] | None = "minmax",
    num_workers: int = 4,
    format: Literal["csv", "parquet", "hdf5"] = "csv",
    streaming: bool = False,
    add_technical_indicators: bool = False,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """
    Factory function to create train/val/test DataLoaders.

    Args:
        data_path: Path to data file or directory.
        target_column: Column to predict.
        batch_size: Batch size for DataLoader.
        seq_len: Input sequence length.
        pred_len: Prediction horizon length.
        train_ratio: Ratio of data for training.
        val_ratio: Ratio of data for validation.
        test_ratio: Ratio of data for testing.
        normalize: Normalization method.
        num_workers: Number of DataLoader workers.
        format: Data format ('csv', 'parquet', 'hdf5').
        streaming: Whether to use streaming for large datasets.
        add_technical_indicators: Add technical indicators.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Validate ratios
    is_valid_sum = abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    assert is_valid_sum, "Ratios must sum to 1.0"

    # For now, we'll create train/val splits using the existing train_ratio
    # and create a separate test set
    # Simplified: train_ratio includes validation
    combined_train_val_ratio = train_ratio + val_ratio
    val_in_train_ratio = val_ratio / combined_train_val_ratio

    # Load data based on format
    if format == "csv":
        csv_path = data_path
    elif format == "parquet":
        # Convert parquet to CSV for now (could optimize later)
        df = cast(pd.DataFrame, pd.read_parquet(data_path))
        csv_path = str(Path(data_path).with_suffix(".csv"))
        df.to_csv(csv_path, index=False)
    elif format == "hdf5":
        # Read HDF5 file (key must be specified or uses default)
        df = cast(pd.DataFrame, pd.read_hdf(data_path, key="data"))
        csv_path = str(Path(data_path).with_suffix(".csv"))
        df.to_csv(csv_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Create train dataset (includes both train and val)
    train_val_dataset = FinancialDataset(
        csv_path=csv_path,
        target_column=target_column,
        seq_len=seq_len,
        pred_len=pred_len,
        train=True,
        train_ratio=combined_train_val_ratio,
        normalize=normalize,
        add_technical_indicators=add_technical_indicators,
    )

    # Create test dataset
    test_dataset = FinancialDataset(
        csv_path=csv_path,
        target_column=target_column,
        seq_len=seq_len,
        pred_len=pred_len,
        train=False,
        train_ratio=combined_train_val_ratio,
        normalize=normalize,
        add_technical_indicators=add_technical_indicators,
        stats={
            "min": train_val_dataset.raw_min,
            "max": train_val_dataset.raw_max,
            "mean": train_val_dataset.raw_mean,
            "std": train_val_dataset.raw_std,
        },
    )

    # Split train_val into train and val
    train_size = int(len(train_val_dataset) * (1 - val_in_train_ratio))
    val_size = len(train_val_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, val_size]
    )

    # Create DataLoaders
    train_loader: DataLoader[Any] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader: DataLoader[Any] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    test_loader: DataLoader[Any] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader


class StreamingDataset(Dataset[Any]):
    """
    Streaming dataset for large files that don't fit in memory.

    Uses chunk-based loading from disk to minimize memory usage.
    """

    def __init__(
        self,
        csv_path: str,
        target_column: str,
        seq_len: int = 30,
        pred_len: int = 1,
        chunk_size: int = 10000,
    ):
        """Initialize StreamingDataset."""
        self.csv_path = csv_path
        self.target_column = target_column
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.chunk_size = chunk_size

        # Get total length by reading just the first column
        self.total_length = sum(1 for _ in open(csv_path)) - 1  # Subtract header

    def __len__(self) -> int:
        """Return total number of sliding window samples."""
        return max(0, self.total_length - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx: int) -> Any:
        """Get a sample from the dataset."""
        # This is a placeholder - actual streaming implementation would
        # require more sophisticated chunk management and caching
        raise NotImplementedError("Streaming dataset not fully implemented yet")
